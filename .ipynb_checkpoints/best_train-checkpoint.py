import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import os
import matplotlib.pyplot as plt
from collections import deque
from minichess_env import MiniChessEnv

# =========================
# 0) 参数区（重点！）
# =========================
BATCH_SIZE = 128
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
LR = 1e-4
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
NUM_EPISODES = 30000

# 防和核弹参数
MAX_TURNS_PER_GAME = 60                 # 60步还没赢 → 强制巨负
FORCE_LOSS_REWARD   = -30.0             # 拖延 = 死罪
WIN_REWARD          = 10.0
LOSE_REWARD         = -10.0
MATERIAL_SCALE      = 0.5
TOPK                = 5
TOPK_RANDOM_P       = 0.05
REPEAT_LIMIT        = 3                 # 三次重复局面也直接判负
REPEAT_PENALTY      = -8.0

# 评估
EVAL_EVERY    = 1000
EVAL_GAMES    = 50
EVAL_MAX_TURNS = 100
AUTO_SAVE_90PLUS = True
SAVE_THRESHOLD   = 90.0

# =========================
# 1) 网络
# =========================
class MiniChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64 * 5 * 5, 256)
        self.fc2   = nn.Linear(256, 625)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def board_to_tensor(board, device):
    x = np.zeros((13, 5, 5), dtype=np.float32)
    for r in range(5):
        for c in range(5):
            p = int(board[r, c])
            if p != 0:
                x[p, r, c] = 1.0
    return torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)

# =========================
# 2) 工具函数
# =========================
PIECE_VALUES = {0:0, 1:1, 2:3, 3:3, 4:5, 5:9, 6:100}

def get_material_score(board, player=1):
    score = my_king = opp_king = 0
    my_k_id  = 12 if player == 1 else 6
    opp_k_id = 6  if player == 1 else 12
    for r in range(5):
        for c in range(5):
            p = int(board[r, c])
            if p == 0: continue
            if p == my_k_id:  my_king = 1
            if p == opp_k_id: opp_king = 1
            val = PIECE_VALUES[p if p <= 6 else p-6]
            if player == 1:
                score += val if p >= 7 else -val
            else:
                score += val if p <= 6 else -val
    return score, bool(my_king), bool(opp_king)

def state_key(board, current_player):
    return board.tobytes() + bytes([1 if current_player == 1 else 0])

# =========================
# 3) 选动作（带 top-k 随机）
# =========================
def select_action(net, state, legals, device, topk_p=TOPK_RANDOM_P):
    with torch.no_grad():
        q = net(state)
        mask = torch.full((1, 625), -float('inf'), device=device)
        mask[0, legals] = 0
        q_masked = q + mask

        if random.random() < topk_p and len(legals) > 1:
            k = min(TOPK, len(legals))
            _, idx = torch.topk(q_masked.flatten(), k)
            return int(idx[torch.randint(0, k, (1,))].item())
        else:
            return int(q_masked.argmax(1).item())

# =========================
# 4) 评估
# =========================
def eval_vs_random(net, device, games=50, max_turns=100):
    wins = losses = draws = 0
    e = MiniChessEnv()
    plt.close(e.fig if hasattr(e, 'fig') else None)
    for _ in range(games):
        board, _ = e.reset()
        done = False
        turns = 0
        player = 1
        while not done and turns < max_turns:
            legals = e.get_legal_moves()
            if not legals:
                wins if player == -1 else losses
                wins += player == -1
                losses += player == 1
                break

            if player == 1:
                s = board_to_tensor(board, device)
                a = select_action(net, s, legals, device, topk_p=0.03)
            else:
                a = random.choice(legals)

            board, _, _, _, _ = e.step(a)
            player *= -1
            turns += 1

            _, my_k, opp_k = get_material_score(board, player=1)
            if not opp_k: wins += 1;   done = True
            if not my_k:  losses += 1; done = True

        if not done: draws += 1

    return wins/games*100, draws/games*100, wins, losses, draws

# =========================
# 5) 主训练（核弹版）
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

policy_net = MiniChessNet().to(device)
target_net = MiniChessNet().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()
memory = deque(maxlen=MEMORY_SIZE)
env = MiniChessEnv()
if hasattr(env, 'fig'): plt.close(env.fig)

stats = {"rewards": [], "losses": [], "eval": []}
steps_done = 0
best_winrate = -1.0

try:
    for episode in range(NUM_EPISODES):
        board, _ = env.reset()
        env.move_log = []
        seen = {}                              # 重复局面检测
        state_tensor = board_to_tensor(board, device)
        total_reward = 0.0
        done = False
        turn_count = 0                         # 全局步数计数

        while not done and turn_count < MAX_TURNS_PER_GAME:
            turn_count += 1

            # 重复局面检测
            key = state_key(board, env.current_player)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= REPEAT_LIMIT:
                reward = REPEAT_PENALTY
                done = True
                total_reward += reward
                break

            legals = env.get_legal_moves()
            if not legals:
                total_reward += LOSE_REWARD
                done = True
                break

            score_before, _, _ = get_material_score(board, player=1)

            # epsilon
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
            steps_done += 1

            if random.random() > eps:
                action = select_action(policy_net, state_tensor, legals, device)
            else:
                # 探索阶段也稍微偏好吃子
                caps = [a for a in legals if board[a%25//5, a%25%5] != 0]
                action = random.choice(caps if caps and random.random()<0.7 else legals)

            # 我方走棋
            board, _, _, _, _ = env.step(action)

            # 我方立即获胜？
            opp_legals = env.get_legal_moves()
            _, _, opp_king = get_material_score(board, player=1)
            if not opp_king or not opp_legals:
                early_bonus = max(0, (MAX_TURNS_PER_GAME - turn_count) * 0.4)
                reward = min(WIN_REWARD + early_bonus, 30.0)
                memory.append((state_tensor, torch.tensor([[action]], device=device, dtype=torch.long),
                               torch.tensor([reward], device=device), None, legals))
                total_reward += reward
                done = True
                break

            # 对手随机走一步
            opp_action = random.choice(opp_legals)
            board, _, _, _, _ = env.step(opp_action)
            turn_count += 1

            # 【核弹1】全局步数到60直接巨负
            if turn_count >= MAX_TURNS_PER_GAME:
                reward = FORCE_LOSS_REWARD
                next_state = None
                done = True
            else:
                score_after, my_king, _ = get_material_score(board, player=1)
                my_next_legals = env.get_legal_moves()

                if not my_king or not my_next_legals:
                    reward = LOSE_REWARD
                    next_state = None
                    done = True
                else:
                    reward = (score_after - score_before) * MATERIAL_SCALE
                    # 【核弹2】指数步数惩罚
                    reward -= 0.008 * (turn_count ** 1.5)
                    next_state = board_to_tensor(board, device)

            memory.append((state_tensor, torch.tensor([[action]], device=device, dtype=torch.long),
                           torch.tensor([reward], device=device), next_state, legals))

            state_tensor = next_state if next_state is not None else state_tensor
            total_reward += float(reward)

            # 学习
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, _ = zip(*batch)

                states = torch.cat(states)
                actions = torch.cat(actions)
                rewards = torch.cat(rewards)

                q_values = policy_net(states).gather(1, actions).squeeze(1)

                next_q = torch.zeros(BATCH_SIZE, device=device)
                non_final = [ns is not None for ns in next_states]
                if any(non_final):
                    ns_batch = torch.cat([ns for ns in next_states if ns is not None])
                    next_q[non_final] = target_net(ns_batch).max(1)[0].detach()

                expected = rewards + GAMMA * next_q
                loss = criterion(q_values, expected)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                stats["losses"].append(loss.item())

        # target net 更新
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        stats["rewards"].append(float(total_reward))

        # 评估 + 保存
        if episode % EVAL_EVERY == 0 and episode > 0:
            policy_net.eval()
            wr, dr, w, l, d = eval_vs_random(policy_net, device, EVAL_GAMES, EVAL_MAX_TURNS)
            policy_net.train()
            print(f"Ep {episode:5d} | eps {eps:.3f} | WR {wr:5.1f}% | Draw {dr:5.1f}% | W/L/D {w}/{l}/{d}")

            stats["eval"].append({"episode": episode, "winrate": wr, "drawrate": dr})

            if wr > best_winrate:
                best_winrate = wr
                torch.save(policy_net.state_dict(), "best_model.pth")
                torch.save(policy_net.state_dict(),
                           f"ckpt_ep{episode}_wr{wr:.1f}_dr{dr:.1f}.pth")
                print("New best! Saved.")

            if AUTO_SAVE_90PLUS and wr >= SAVE_THRESHOLD:
                torch.save(policy_net.state_dict(), "best_model_90plus.pth")
                print("90%+! Saved best_model_90plus.pth")

            with open("training_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    print("Final saving...")
    torch.save(policy_net.state_dict(), "best_model.pth")
    with open("training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    plt.close('all')
    print("All done! best_model.pth + training_stats.json")