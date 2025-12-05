import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from minichess_env import MiniChessEnv
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 10000

# =====================
# 0) 关键超参数（优先别乱改）
# =====================
NUM_EPISODES = 30000
BATCH_SIZE   = 128
MEMORY_SIZE  = 50000
LR           = 1e-4
GAMMA        = 0.98
TARGET_UPDATE = 200  # 稳一点，别太频繁

# 减少 draw 的核心开关
TOPK = 5
P_TOPK_TRAIN = 0.06     # 训练 exploitation 中 6% 概率 top-k 随机，破循环
P_TOPK_EVAL  = 0.03     # 评估更保守
REPEAT_LIMIT = 3        # 同一局面出现3次 -> 强惩罚并终止
REPEAT_PENALTY = -8.0

# “无进展”提前判和（训练上把和棋当坏事）
NO_CAPTURE_LIMIT = 20   # 连续20个回合都没吃子 -> 判和并惩罚（训练用）
NO_CAPTURE_PENALTY = -6.0

# 奖励缩放（别用几千几万，Q会炸）
WIN_REWARD  = 10.0
LOSE_REWARD = -10.0
MATERIAL_SCALE = 0.6
STEP_PENALTY = 0.05
MOBILITY_SCALE = 0.03   # 小一点就好

# 评估
EVAL_EVERY = 1000
EVAL_GAMES = 50
MAX_TURNS  = 100

# 达标保存
SAVE_WR_THRESHOLD = 90.0
SAVE_SELF_DRAW_THRESHOLD = 70.0  # self-play draw <=70% 就算“明显改善”（你也可以调更严，比如 50）

# =====================
# 1) 网络
# =====================
class MiniChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 625)

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
            x[p, r, c] = 1.0
    return torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)

# =====================
# 2) 评分/终局帮助函数
# =====================
PIECE_VALUES = {0:0, 1:1, 2:3, 3:3, 4:5, 5:9, 6:100}

def material_score_gold(board):
    """
    从 Gold(玩家=1) 视角：Gold(>=7)加分，Silver(<=6)减分
    同时返回 Gold王/ Silver王 是否还活着
    """
    score = 0
    gold_king_alive = False
    silver_king_alive = False
    for r in range(5):
        for c in range(5):
            p = int(board[r, c])
            if p == 0:
                continue
            if p == 12:  # Gold king
                gold_king_alive = True
            if p == 6:   # Silver king
                silver_king_alive = True
            base = p if p <= 6 else p - 6
            val = PIECE_VALUES.get(base, 0)
            score += val if p >= 7 else -val
    return score, gold_king_alive, silver_king_alive

def state_key(board, current_player):
    # 局面 + 到谁走（否则同棋盘不同回合会被误判重复）
    return board.tobytes() + bytes([1 if current_player == 1 else 0])

def infer_capture(prev_board, next_board):
    # 简单判断：棋子总数减少就是发生吃子（对 5x5 足够）
    return int(np.count_nonzero(next_board)) < int(np.count_nonzero(prev_board))

# =====================
# 3) 动作选择（mask + argmax / top-k随机）
# =====================
def select_action(policy_net, state_tensor, legal_moves, device, p_topk):
    with torch.no_grad():
        q = policy_net(state_tensor)  # (1,625)
        mask = torch.full((1, 625), -float('inf'), device=device, dtype=torch.float32)
        mask[0, legal_moves] = 0.0
        q_masked = q + mask

        if len(legal_moves) <= 1:
            return int(torch.argmax(q_masked, dim=1).item())

        # 绝大多数走最优，少量 top-k 随机破循环
        if random.random() > p_topk:
            return int(torch.argmax(q_masked, dim=1).item())

        k = min(TOPK, len(legal_moves))
        _, topk_idx = torch.topk(q_masked, k, dim=1)
        j = int(torch.randint(0, k, (1,), device=device).item())
        return int(topk_idx[0, j].item())

# =====================
# 4) 评估：vs Random & Self-play
# =====================
def play_episode(agentA, agentB, max_turns=100):
    """
    agentA 控制 current_player == 1（Gold）
    agentB 控制 current_player == -1（Silver）
    agent: 函数(board, legal_moves, p_topk) -> action
    返回 winner: 1/-1/0(draw), reason
    """
    env = MiniChessEnv()
    if hasattr(env, "fig"):
        plt.close(env.fig)
    board, _ = env.reset()

    for t in range(max_turns):
        legal = env.get_legal_moves()
        if not legal:
            winner = -env.current_player
            return winner, "No legal moves"

        if env.current_player == 1:
            a = agentA(board, legal)
        else:
            a = agentB(board, legal)

        if a not in legal:
            winner = -env.current_player
            return winner, "Illegal move"

        prev = board.copy()
        board, _, _, _, _ = env.step(a)

        # 王被吃也强制视为终局（环境不判，我们自己判）
        _, gold_k, silver_k = material_score_gold(board)
        if not silver_k:
            return 1, "Silver king captured"
        if not gold_k:
            return -1, "Gold king captured"

    return 0, "Draw: max turns"

def eval_vs_random(policy_net, device, games=50):
    wins = losses = draws = 0

    def agent_model(board, legal):
        s = board_to_tensor(board, device)
        a = select_action(policy_net, s, legal, device, p_topk=P_TOPK_EVAL)
        return a

    def agent_random(board, legal):
        return random.choice(legal)

    for _ in range(games):
        w, _ = play_episode(agent_model, agent_random, max_turns=MAX_TURNS)
        if w == 1: wins += 1
        elif w == -1: losses += 1
        else: draws += 1

    wr = wins / games * 100.0
    dr = draws / games * 100.0
    return wr, dr, wins, losses, draws

def eval_self_play(policy_net, device, games=20):
    draws = wins = losses = 0

    def agent_model(board, legal):
        s = board_to_tensor(board, device)
        # self-play 给稍微大一点的随机，否则太容易镜像循环
        a = select_action(policy_net, s, legal, device, p_topk=max(P_TOPK_EVAL, 0.05))
        return a

    for _ in range(games):
        w, _ = play_episode(agent_model, agent_model, max_turns=MAX_TURNS)
        if w == 0: draws += 1
        elif w == 1: wins += 1
        else: losses += 1

    self_draw = draws / games * 100.0
    return self_draw, wins, losses, draws

# =====================
# 5) 训练主循环
# =====================
def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    policy_net = MiniChessNet().to(device)
    target_net = MiniChessNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()
    memory = deque(maxlen=MEMORY_SIZE)

    env = MiniChessEnv()
    if hasattr(env, "fig"):
        plt.close(env.fig)

    stats = {"rewards": [], "losses": [], "eval": []}
    best_score = -1e9

    for ep in range(NUM_EPISODES):
        board, _ = env.reset()
        env.move_log = []

        seen = {}
        no_capture_cnt = 0

        total_reward = 0.0
        done = False

        while not done:
            # ===== 轮到 Gold（我们训练的主体）=====
            if env.current_player != 1:
                # 保险：如果环境出现异常，强行让 random 走到 Gold 回合
                legal = env.get_legal_moves()
                if not legal:
                    total_reward += WIN_REWARD
                    break
                board, _, _, _, _ = env.step(random.choice(legal))
                continue

            # 重复检测（回合开始）
            k0 = state_key(board, env.current_player)
            seen[k0] = seen.get(k0, 0) + 1
            if seen[k0] >= REPEAT_LIMIT:
                total_reward += REPEAT_PENALTY
                done = True
                break

            legal_moves = env.get_legal_moves()
            if not legal_moves:
                total_reward += LOSE_REWARD
                done = True
                break

            score_start, gold_k0, silver_k0 = material_score_gold(board)
            if not gold_k0:
                total_reward += LOSE_REWARD
                done = True
                break
            if not silver_k0:
                total_reward += WIN_REWARD
                done = True
                break

            state_tensor = board_to_tensor(board, device)

            # epsilon：用更标准的线性/指数都行，这里保持简单
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-ep / (EPS_DECAY / 1000.0))

            if random.random() > eps:
                action = select_action(policy_net, state_tensor, legal_moves, device, p_topk=P_TOPK_TRAIN)
            else:
                # 探索时偏好吃子（更快结束）
                cap = []
                for a in legal_moves:
                    to = a % 25
                    tr, tc = divmod(to, 5)
                    if int(board[tr, tc]) != 0:
                        cap.append(a)
                action = random.choice(cap) if cap and random.random() < 0.7 else random.choice(legal_moves)

            prev_board = board.copy()
            board, _, _, _, _ = env.step(action)

            captured = infer_capture(prev_board, board)
            no_capture_cnt = 0 if captured else (no_capture_cnt + 1)

            # ===== 终局检查（我方走完）=====
            opp_legal = env.get_legal_moves()
            score_mid, gold_k1, silver_k1 = material_score_gold(board)
            if (not silver_k1) or (not opp_legal):
                r = WIN_REWARD
                memory.append((
                    state_tensor,
                    torch.tensor([[action]], device=device, dtype=torch.int64),
                    torch.tensor([r], device=device, dtype=torch.float32),
                    None
                ))
                total_reward += r
                done = True
                break

            # ===== 对手 Silver 随机走一步 =====
            opp_action = random.choice(opp_legal)
            prev2 = board.copy()
            board, _, _, _, _ = env.step(opp_action)

            captured2 = infer_capture(prev2, board)
            no_capture_cnt = 0 if captured2 else (no_capture_cnt + 1)

            # ===== 结算奖励（一个“回合”：我走+对手走）=====
            score_end, gold_k2, silver_k2 = material_score_gold(board)

            # mobility 奖励：更快把对手逼到没法走
            my_legal_next = env.get_legal_moves()
            # 注意：此时应该轮到 Gold（env.current_player==1），所以 my_legal_next 是 Gold 的步
            # 为了估计对手 mobility，用“假走到对手回合”很麻烦，这里用轻量近似：只奖励“我有更多步”
            mobility_bonus = MOBILITY_SCALE * (len(my_legal_next) - 10)

            if (not gold_k2) or (not my_legal_next):
                r = LOSE_REWARD
                next_state = None
                done = True
            else:
                r = (score_end - score_start) * MATERIAL_SCALE
                r -= STEP_PENALTY
                r += mobility_bonus
                next_state = board_to_tensor(board, device)

            # 无进展判和（训练上把 draw 当坏事）
            if no_capture_cnt >= NO_CAPTURE_LIMIT:
                r += NO_CAPTURE_PENALTY
                next_state = None
                done = True

            # 对手走完后重复检测（最关键：防循环）
            k1 = state_key(board, env.current_player)
            seen[k1] = seen.get(k1, 0) + 1
            if seen[k1] >= REPEAT_LIMIT:
                r = REPEAT_PENALTY
                next_state = None
                done = True

            memory.append((
                state_tensor,
                torch.tensor([[action]], device=device, dtype=torch.int64),
                torch.tensor([r], device=device, dtype=torch.float32),
                next_state
            ))
            total_reward += float(r)

            # ===== 学习 =====
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_s, b_a, b_r, b_ns = zip(*batch)

                b_s = torch.cat(b_s)
                b_a = torch.cat(b_a)
                b_r = torch.cat(b_r).float()

                q_sa = policy_net(b_s).gather(1, b_a).squeeze(1)

                non_final_mask = torch.tensor([ns is not None for ns in b_ns], device=device, dtype=torch.bool)
                next_vals = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32)
                if non_final_mask.any():
                    non_final_next_states = torch.cat([ns for ns in b_ns if ns is not None])
                    with torch.no_grad():
                        next_vals[non_final_mask] = target_net(non_final_next_states).max(1)[0]

                target = b_r + GAMMA * next_vals
                loss = criterion(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                stats["losses"].append(float(loss.item()))

        # target 更新
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        stats["rewards"].append(float(total_reward))

        # ===== 评估 & 自动保存 =====
        if ep % EVAL_EVERY == 0:
            policy_net.eval()
            wr, dr, w, l, d = eval_vs_random(policy_net, device, games=EVAL_GAMES)
            self_draw, sw, sl, sd = eval_self_play(policy_net, device, games=20)
            policy_net.train()

            # 综合分：你要 win高+draw低
            score = wr - 1.2 * dr - 1.5 * self_draw
            print(f"Ep {ep:5d} | EpR {total_reward:7.2f} | "
                  f"WR(vsRand) {wr:5.1f}% DR {dr:5.1f}% | SelfDraw {self_draw:5.1f}% | score {score:7.2f}")

            stats["eval"].append({
                "ep": ep,
                "wr_vs_random": wr, "dr_vs_random": dr, "W": w, "L": l, "D": d,
                "self_draw": self_draw, "self_W": sw, "self_L": sl, "self_D": sd,
                "score": score
            })

            # 保存“最好”的模型（综合考虑）
            if score > best_score:
                best_score = score
                torch.save(policy_net.state_dict(), "cuda3best_model.pth")
                torch.save(policy_net.state_dict(), f"ckpt_ep{ep}_wr{wr:.1f}_dr{dr:.1f}_sd{self_draw:.1f}.pth")
                print("💾 Improved -> saved best_model.pth")

            # 你的硬目标：vs random 90+ 且 self draw 不那么离谱
            if wr >= SAVE_WR_THRESHOLD and self_draw <= SAVE_SELF_DRAW_THRESHOLD:
                torch.save(policy_net.state_dict(), "best_model_90plus.pth")
                print("🏁 Saved best_model_90plus.pth (WR>=90 and self-draw improved)")

            with open("training_stats.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    # 结束保存
    torch.save(policy_net.state_dict(), "final_model.pth")
    print("✅ Done. Saved final_model.pth")

if __name__ == "__main__":
    main()
