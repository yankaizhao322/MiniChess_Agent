# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import random
# # import json
# # import matplotlib.pyplot as plt 
# # from collections import deque
# # from minichess_env import MiniChessEnv

# # class MiniChessNet(nn.Module):
# #     def __init__(self):
# #         super(MiniChessNet, self).__init__()

# #         self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
# #         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

# #         self.fc1 = nn.Linear(64 * 5 * 5, 256) 
# #         self.fc2 = nn.Linear(256, 625)

# #     def forward(self, x):
# #         x = torch.relu(self.conv1(x))
# #         x = torch.relu(self.conv2(x))
# #         x = x.view(x.size(0), -1)
# #         x = torch.relu(self.fc1(x))
# #         return self.fc2(x)

# # def board_to_tensor(board, device):
# #     x = np.zeros((13, 5, 5), dtype=np.float32)
# #     for r in range(5):
# #         for c in range(5):
# #             piece_idx = board[r, c]
# #             x[piece_idx, r, c] = 1.0
# #     return torch.tensor(x, device=device).unsqueeze(0)

# # BATCH_SIZE = 128     # 小 Batch 更新更频繁
# # GAMMA = 0.95
# # EPS_START = 1.0
# # EPS_END = 0.05
# # EPS_DECAY = 10000    # 【关键】2000局内就把探索降到底，逼它快速学会
# # LR = 1e-4           # 【关键】高学习率，学得快
# # TARGET_UPDATE = 10
# # MEMORY_SIZE = 20000
# # NUM_EPISODES = 30000 

# # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# # print(f"Turbo Training Mode | Device: {device}")

# # policy_net = MiniChessNet().to(device)
# # target_net = MiniChessNet().to(device)
# # target_net.load_state_dict(policy_net.state_dict())
# # target_net.eval()

# # optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# # criterion = nn.SmoothL1Loss()
# # memory = deque(maxlen=MEMORY_SIZE)
# # env = MiniChessEnv()
# # plt.close(env.fig)

# # PIECE_VALUES = {0:0, 1:1, 2:3, 3:3, 4:5, 5:9, 6:100}

# # def get_material_score(board, player):
# #     score = 0
# #     my_king = False
# #     opp_king = False
# #     my_k_id = 12 if player == 1 else 6
# #     opp_k_id = 6 if player == 1 else 12

# #     for r in range(5):
# #         for c in range(5):
# #             p = board[r, c]
# #             if p == 0: continue
# #             if p == my_k_id: my_king = True
# #             if p == opp_k_id: opp_king = True
# #             val = PIECE_VALUES[p if p <=6 else p-6]
# #             if player == 1: 
# #                 if p >= 7: score += val
# #                 else: score -= val
# #             else: 
# #                 if p <= 6: score += val
# #                 else: score -= val
# #     return score, my_king, opp_king

# # def eval_win_rate(agent_net, episodes=20):
# #     wins = 0
# #     test_env = MiniChessEnv()
# #     plt.close(test_env.fig)
    
# #     for _ in range(episodes):
# #         board, _ = test_env.reset()
# #         done = False
# #         turns = 0
# #         player = 1
# #         while not done and turns < 50:
# #             legal = test_env.get_legal_moves()
# #             if not legal: break
            
# #             if player == 1: # AI
# #                 with torch.no_grad():
# #                     t = board_to_tensor(board, device)
# #                     q = agent_net(t)
# #                     mask = torch.full((1, 625), -float('inf'), device=device)
# #                     mask[0, legal] = 0
# #                     action = (q + mask).max(1)[1].item()
# #             else: # Random Opponent
# #                 action = random.choice(legal)
                
# #             board, _, _, _, _ = test_env.step(action)
# #             player *= -1
# #             turns += 1
            
# #             # Check win condition for AI (Player 1)
# #             # After step, player flipped. So if player is now -1, AI just moved.
# #             # Check if Opponent (-1) is dead
# #             opp_legal = test_env.get_legal_moves()
# #             _, _, opp_k = get_material_score(board, 1)
            
# #             if (not opp_k or not opp_legal) and player == -1:
# #                 wins += 1
# #                 done = True
                
# #     return (wins / episodes) * 100

# # stats = {"rewards": [], "losses": []}
# # steps_done = 0

# # try:
# #     for i_episode in range(NUM_EPISODES):
# #         board, _ = env.reset()
# #         env.move_log = [] 
# #         state_tensor = board_to_tensor(board, device)
# #         total_reward = 0
# #         player = 1 
# #         done = False
        
# #         while not done:
# #             # --- My Move ---
# #             legal_moves = env.get_legal_moves()
# #             if not legal_moves: break 

# #             score_start, _, _ = get_material_score(board, player)

# #             sample = random.random()
# #             eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
# #             steps_done += 1
            
# #             if sample > eps_threshold:
# #                 with torch.no_grad():
# #                     q = policy_net(state_tensor)
# #                     mask = torch.full((1, 625), -float('inf'), device=device)
# #                     mask[0, legal_moves] = 0 
# #                     action_idx = (q + mask).max(1)[1].item()
# #             else:
# #                 action_idx = random.choice(legal_moves)

# #             board, _, _, _, _ = env.step(action_idx)
            
# #             opp_legal = env.get_legal_moves()
# #             _, _, opp_king_alive = get_material_score(board, player)
            
# #             if not opp_king_alive or not opp_legal:
# #                 reward = 2000 
# #                 memory.append((state_tensor, torch.tensor([[action_idx]], device=device), 
# #                                torch.tensor([reward], device=device), None, legal_moves))
# #                 total_reward += reward
# #                 done = True
# #                 break
            
# #             # --- Opponent Move (Random) ---
# #             opp_action = random.choice(opp_legal)
# #             board, _, _, _, _ = env.step(opp_action)
            
# #             # --- Check Loss/Net Score ---
# #             score_end, my_king_alive, _ = get_material_score(board, player)
# #             my_next_legal = env.get_legal_moves()
            
# #             if not my_king_alive or not my_next_legal:
# #                 reward = -1000 
# #                 next_state_tensor = None
# #                 done = True
# #             else:
# #                 reward = (score_end - score_start) * 50 
                
# #                 reward -= 5 
                
# #                 next_state_tensor = board_to_tensor(board, device)

# #             memory.append((state_tensor, torch.tensor([[action_idx]], device=device), 
# #                            torch.tensor([reward], device=device), next_state_tensor, legal_moves))
            
# #             state_tensor = next_state_tensor if next_state_tensor is not None else state_tensor
# #             total_reward += reward
            
# #             if len(memory) > BATCH_SIZE:
# #                 transitions = random.sample(memory, BATCH_SIZE)
# #                 batch_state, batch_action, batch_reward, batch_next_state, _ = zip(*transitions)
                
# #                 batch_state = torch.cat(batch_state)
# #                 batch_action = torch.cat(batch_action)
# #                 batch_reward = torch.cat(batch_reward)
                
# #                 current_q = policy_net(batch_state).gather(1, batch_action)
# #                 non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=device, dtype=torch.bool)
# #                 non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
                
# #                 next_state_values = torch.zeros(BATCH_SIZE, device=device)
# #                 if len(non_final_next_states) > 0:
# #                     next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                
# #                 expected_q = batch_reward + (GAMMA * next_state_values)
# #                 loss = criterion(current_q, expected_q.unsqueeze(1))
                
# #                 optimizer.zero_grad()
# #                 loss.backward()
# #                 torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
# #                 optimizer.step()
# #                 stats["losses"].append(loss.item())

# #         if i_episode % TARGET_UPDATE == 0:
# #             target_net.load_state_dict(policy_net.state_dict())
            
# #         stats["rewards"].append(total_reward)
        
# #         if i_episode % 1000 == 0:
# #             win_rate = eval_win_rate(policy_net)
# #             print(f"Ep {i_episode} | Eps: {eps_threshold:.2f} | Win Rate (vs Random): {win_rate}%")
            
# #             if win_rate >= 85:
# #                 print("Saving:")
# #                 torch.save(policy_net.state_dict(), "cuda2best_model.pth")

# # except KeyboardInterrupt:
# #     print("Stopped.")
# # finally:
# #     print("Final Save...")
# #     torch.save(policy_net.state_dict(), "best_model.pth")
# #     with open("training_stats.json", "w") as f:
# #         json.dump(stats, f)
# #     plt.close('all')
# # # import os
# # # import json
# # # import random
# # # from collections import deque, defaultdict

# # # import numpy as np
# # # import torch
# # # import torch.nn as nn
# # # import torch.optim as optim
# # # import matplotlib.pyplot as plt

# # # from minichess_env import MiniChessEnv


# # # # =========================
# # # # 0) 固定随机种子（可复现）
# # # # =========================
# # # SEED = 0
# # # random.seed(SEED)
# # # np.random.seed(SEED)
# # # torch.manual_seed(SEED)


# # # # =========================
# # # # 1) 轻量级网络（与你的思路一致）
# # # # =========================
# # # class MiniChessNet(nn.Module):
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
# # #         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# # #         self.fc1 = nn.Linear(64 * 5 * 5, 256)
# # #         self.fc2 = nn.Linear(256, 625)

# # #     def forward(self, x):
# # #         x = torch.relu(self.conv1(x))
# # #         x = torch.relu(self.conv2(x))
# # #         x = x.reshape(x.size(0), -1)
# # #         x = torch.relu(self.fc1(x))
# # #         return self.fc2(x)


# # # def board_to_tensor(board: np.ndarray, device: torch.device) -> torch.Tensor:
# # #     # one-hot: (13,5,5) float32
# # #     x = np.zeros((13, 5, 5), dtype=np.float32)
# # #     for r in range(5):
# # #         for c in range(5):
# # #             p = int(board[r, c])
# # #             x[p, r, c] = 1.0
# # #     return torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)


# # # # =========================
# # # # 2) Canonicalize：把 “当前走棋方” 统一映射为同一视角（强烈建议）
# # # #    这样一个网络能学 Gold/Silver 两边，减少 self-play 对称循环
# # # # =========================
# # # _SWAP = np.zeros(13, dtype=np.int8)
# # # for p in range(1, 7):
# # #     _SWAP[p] = p + 6
# # #     _SWAP[p + 6] = p

# # # def canon_board(board: np.ndarray, player: int) -> np.ndarray:
# # #     """
# # #     返回 canonical board：让当前走子方视角总是“Gold视角”
# # #     player=1: 原样
# # #     player=-1: 旋转180度 + 交换阵营编码
# # #     """
# # #     if player == 1:
# # #         return board
# # #     b = np.rot90(board, 2).copy()
# # #     return _SWAP[b]

# # # def canon_action(action: int, player: int) -> int:
# # #     if player == 1:
# # #         return action
# # #     from_idx = action // 25
# # #     to_idx = action % 25
# # #     from_idx = 24 - from_idx
# # #     to_idx = 24 - to_idx
# # #     return from_idx * 25 + to_idx

# # # def uncanon_action(action_c: int, player: int) -> int:
# # #     # same mapping (involution)
# # #     return canon_action(action_c, player)


# # # # =========================
# # # # 3) Reward shaping：防止全 draw 的关键
# # # #    - 用材质变化（小尺度）来引导
# # # #    - 每步小惩罚，推动尽快分胜负
# # # #    - 重复局面惩罚，减少循环（你们评分看 draw 少）
# # # # =========================
# # # PIECE_VAL = {0:0, 1:1, 2:3, 3:3, 4:5, 5:9, 6:30}

# # # def material(board: np.ndarray, player: int) -> float:
# # #     """从 player 视角：我方材质-对方材质"""
# # #     s = 0.0
# # #     me_is_gold = (player == 1)
# # #     for r in range(5):
# # #         for c in range(5):
# # #             p = int(board[r, c])
# # #             if p == 0:
# # #                 continue
# # #             base = p if p <= 6 else p - 6
# # #             v = PIECE_VAL.get(base, 0)
# # #             is_gold_piece = (p >= 7)
# # #             s += v if (is_gold_piece == me_is_gold) else -v
# # #     return float(s)

# # # def state_key(board: np.ndarray, player: int) -> bytes:
# # #     # 记录局面 + 到谁走，检测重复
# # #     return board.tobytes() + bytes([1 if player == 1 else 0])


# # # # =========================
# # # # 4) Replay Buffer
# # # # =========================
# # # class Replay:
# # #     def __init__(self, cap: int):
# # #         self.buf = deque(maxlen=cap)

# # #     def push(self, s_c, a_c, r, s2_c, done):
# # #         # 全放 CPU numpy，省显存
# # #         self.buf.append((s_c, int(a_c), float(r), s2_c, bool(done)))

# # #     def sample(self, n: int):
# # #         batch = random.sample(self.buf, n)
# # #         s, a, r, s2, d = zip(*batch)
# # #         return s, a, r, s2, d

# # #     def __len__(self):
# # #         return len(self.buf)


# # # # =========================
# # # # 5) 训练/评估配置（你关心的“>90 自动保存”在这里）
# # # # =========================
# # # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print(f"🚀 Device: {DEVICE}")

# # # NUM_EPISODES = 30000
# # # MAX_TURNS = 100  # match_runner 通常 100 步就 draw，所以训练也对齐

# # # BATCH_SIZE = 128
# # # REPLAY_SIZE = 60000
# # # WARMUP_STEPS = 2000

# # # GAMMA = 0.98
# # # LR = 1e-4
# # # GRAD_CLIP = 1.0

# # # EPS_START = 1.0
# # # EPS_END = 0.05
# # # EPS_DECAY_STEPS = 40000

# # # TARGET_UPDATE_STEPS = 800  # hard copy target

# # # # repetition draw discourage
# # # REPEAT_LIMIT = 3
# # # REPEAT_PENALTY = -0.2

# # # # 每隔多少局评估一次
# # # EVAL_EVERY = 1000
# # # GAMES_EVAL = 30

# # # # >90 自动保存阈值（按 vs Random 胜率）
# # # AUTO_SAVE = True
# # # SAVE_THRESHOLD = 90.0

# # # # 额外：你也可以同时约束 self-play draw
# # # MAX_SELF_DRAW_FOR_SUBMIT = 70.0  # 想严格点就降，例如 50


# # # def epsilon_by_step(step: int) -> float:
# # #     return float(EPS_END + (EPS_START - EPS_END) * np.exp(-step / float(EPS_DECAY_STEPS)))


# # # # =========================
# # # # 6) 评估：vs Random / self-play draw%
# # # # =========================
# # # def eval_vs_random(net: nn.Module, games=30) -> float:
# # #     wins = 0
# # #     e = MiniChessEnv()
# # #     if hasattr(e, "fig"):
# # #         plt.close(e.fig)

# # #     for _ in range(games):
# # #         board, _ = e.reset()
# # #         player = 1  # our agent is Gold, random is Silver
# # #         for _t in range(100):
# # #             legal = e.get_legal_moves()
# # #             if not legal:
# # #                 # current player has no moves => loses
# # #                 if player == -1:
# # #                     wins += 1
# # #                 break

# # #             if player == 1:
# # #                 b_c = canon_board(board, player)
# # #                 legal_c = [canon_action(a, player) for a in legal]
# # #                 with torch.no_grad():
# # #                     s = board_to_tensor(b_c, DEVICE)
# # #                     q = net(s)[0]
# # #                     mask = torch.full((625,), -1e9, device=DEVICE, dtype=torch.float32)
# # #                     mask[torch.tensor(legal_c, device=DEVICE)] = 0.0
# # #                     a_c = int(torch.argmax(q + mask).item())
# # #                 a = uncanon_action(a_c, player)
# # #                 if a not in legal:
# # #                     a = random.choice(legal)
# # #             else:
# # #                 a = random.choice(legal)

# # #             board, _, _, _, _ = e.step(a)
# # #             player *= -1

# # #     return wins / games * 100.0


# # # def eval_self_play(net: nn.Module, games=20):
# # #     wins = losses = draws = 0
# # #     e = MiniChessEnv()
# # #     if hasattr(e, "fig"):
# # #         plt.close(e.fig)

# # #     for _ in range(games):
# # #         board, _ = e.reset()
# # #         player = 1
# # #         for _t in range(100):
# # #             legal = e.get_legal_moves()
# # #             if not legal:
# # #                 # current player loses
# # #                 if player == 1:
# # #                     losses += 1
# # #                 else:
# # #                     wins += 1
# # #                 break

# # #             b_c = canon_board(board, player)
# # #             legal_c = [canon_action(a, player) for a in legal]
# # #             with torch.no_grad():
# # #                 s = board_to_tensor(b_c, DEVICE)
# # #                 q = net(s)[0]
# # #                 mask = torch.full((625,), -1e9, device=DEVICE, dtype=torch.float32)
# # #                 mask[torch.tensor(legal_c, device=DEVICE)] = 0.0
# # #                 a_c = int(torch.argmax(q + mask).item())
# # #             a = uncanon_action(a_c, player)
# # #             if a not in legal:
# # #                 a = random.choice(legal)

# # #             board, _, _, _, _ = e.step(a)
# # #             player *= -1
# # #         else:
# # #             draws += 1

# # #     wr = wins / games * 100.0
# # #     dr = draws / games * 100.0
# # #     return wr, dr, wins, losses, draws


# # # # =========================
# # # # 7) 初始化模型
# # # # =========================
# # # policy = MiniChessNet().to(DEVICE).float()
# # # target = MiniChessNet().to(DEVICE).float()
# # # target.load_state_dict(policy.state_dict())
# # # target.eval()

# # # opt = optim.AdamW(policy.parameters(), lr=LR)
# # # loss_fn = nn.SmoothL1Loss()

# # # replay = Replay(REPLAY_SIZE)

# # # env = MiniChessEnv()
# # # if hasattr(env, "fig"):
# # #     plt.close(env.fig)

# # # stats = {
# # #     "ep_return": [],
# # #     "loss": [],
# # #     "eval": []  # list of dicts with ep, winrate_random, self_draw
# # # }

# # # global_step = 0
# # # best_wr_random = -1.0
# # # best_self_draw = 100.0

# # # print("🏁 Start training...")

# # # for ep in range(NUM_EPISODES):
# # #     board, _ = env.reset()
# # #     env.move_log = []  # 防内存泄漏
# # #     player = 1
# # #     ep_ret = 0.0

# # #     # repetition counter per episode
# # #     rep = defaultdict(int)

# # #     for turn in range(MAX_TURNS):
# # #         # 当前局面重复次数（用于惩罚循环）
# # #         k = state_key(board, player)
# # #         rep[k] += 1

# # #         legal = env.get_legal_moves()
# # #         if not legal:
# # #             # current player loses, previous player "wins"
# # #             break

# # #         # canonical state/action space
# # #         b_c = canon_board(board, player)
# # #         legal_c = [canon_action(a, player) for a in legal]

# # #         eps = epsilon_by_step(global_step)
# # #         global_step += 1

# # #         # epsilon-greedy（注意：explore 时也尽量选 capture 来减少 draw）
# # #         if random.random() < eps:
# # #             # 60% pick best capture, else random
# # #             def capture_flag(a):
# # #                 to_idx = a % 25
# # #                 tr, tc = divmod(to_idx, 5)
# # #                 return 1 if board[tr, tc] != 0 else 0
# # #             legal_sorted = sorted(legal, key=capture_flag, reverse=True)
# # #             a = legal_sorted[0] if random.random() < 0.6 else random.choice(legal)
# # #             a_c = canon_action(a, player)
# # #         else:
# # #             with torch.no_grad():
# # #                 s = board_to_tensor(b_c, DEVICE)
# # #                 q = policy(s)[0]
# # #                 mask = torch.full((625,), -1e9, device=DEVICE, dtype=torch.float32)
# # #                 mask[torch.tensor(legal_c, device=DEVICE)] = 0.0
# # #                 a_c = int(torch.argmax(q + mask).item())
# # #             a = uncanon_action(a_c, player)
# # #             if a not in legal:
# # #                 a = random.choice(legal)
# # #                 a_c = canon_action(a, player)

# # #         # reward shaping: material delta + step penalty + repetition penalty
# # #         before = material(board, player)

# # #         next_board, _, _, _, _ = env.step(a)
# # #         after = material(next_board, player)

# # #         # 小尺度，避免 Q 发散（你之前爆炸就是尺度太大）
# # #         r = 0.05 * float(np.clip(after - before, -6, 6)) - 0.01

# # #         # 若造成重复局面（下一手），加惩罚
# # #         next_key = state_key(next_board, -player)
# # #         rep[next_key] += 0  # ensure exists
# # #         if rep[next_key] >= REPEAT_LIMIT:
# # #             r += REPEAT_PENALTY

# # #         # terminal: opponent has no legal moves => I win
# # #         env.board = next_board.copy()
# # #         env.current_player = -player
# # #         opp_legal = env.get_legal_moves()
# # #         done = (len(opp_legal) == 0)
# # #         if done:
# # #             r += 1.0  # win bonus

# # #         # store transition in canonical (to-move) form
# # #         s_c_np = b_c.copy()
# # #         s2_c_np = canon_board(next_board, -player).copy()
# # #         replay.push(s_c_np, a_c, r, s2_c_np, done)

# # #         ep_ret += r

# # #         # move on
# # #         board = next_board
# # #         player *= -1

# # #         # learn
# # #         if len(replay) >= max(WARMUP_STEPS, BATCH_SIZE):
# # #             S_np, A_np, R_np, S2_np, D_np = replay.sample(BATCH_SIZE)

# # #             S = torch.cat([board_to_tensor(x, DEVICE) for x in S_np], dim=0)
# # #             A = torch.tensor(A_np, device=DEVICE, dtype=torch.int64).unsqueeze(1)
# # #             Rb = torch.tensor(R_np, device=DEVICE, dtype=torch.float32)
# # #             S2 = torch.cat([board_to_tensor(x, DEVICE) for x in S2_np], dim=0)
# # #             Db = torch.tensor(D_np, device=DEVICE, dtype=torch.float32)

# # #             # Q(s,a)
# # #             q_sa = policy(S).gather(1, A).squeeze(1)

# # #             # Double DQN target
# # #             with torch.no_grad():
# # #                 a2 = torch.argmax(policy(S2), dim=1, keepdim=True)
# # #                 q2 = target(S2).gather(1, a2).squeeze(1)
# # #                 y = Rb + (1.0 - Db) * GAMMA * q2

# # #             loss = loss_fn(q_sa, y)

# # #             opt.zero_grad()
# # #             loss.backward()
# # #             torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
# # #             opt.step()

# # #             stats["loss"].append(float(loss.item()))

# # #         # update target
# # #         if global_step % TARGET_UPDATE_STEPS == 0:
# # #             target.load_state_dict(policy.state_dict())

# # #         if done:
# # #             break

# # #     stats["ep_return"].append(float(ep_ret))

# # #     # 评估 + 自动保存
# # #     if ep % EVAL_EVERY == 0:
# # #         policy.eval()
# # #         wr_random = eval_vs_random(policy, games=GAMES_EVAL)
# # #         wr_self, dr_self, w, l, d = eval_self_play(policy, games=max(10, GAMES_EVAL // 2))
# # #         policy.train()

# # #         stats["eval"].append({
# # #             "ep": int(ep),
# # #             "global_step": int(global_step),
# # #             "eps": float(eps),
# # #             "wr_random": float(wr_random),
# # #             "wr_self": float(wr_self),
# # #             "draw_self": float(dr_self)
# # #         })

# # #         print(f"Ep {ep:5d} | steps {global_step:7d} | eps {eps:.2f} | "
# # #               f"EpR {ep_ret:+.3f} | WR(vsRand) {wr_random:5.1f}% | SelfDraw {dr_self:5.1f}%")

# # #         # 只要“更好”就保存 best_model.pth
# # #         improved = (wr_random > best_wr_random) or (wr_random == best_wr_random and dr_self < best_self_draw)
# # #         if improved:
# # #             best_wr_random = max(best_wr_random, wr_random)
# # #             best_self_draw = min(best_self_draw, dr_self)
# # #             torch.save(policy.state_dict(), "best_model.pth")
# # #             ckpt = f"ckpt_ep{ep}_wr{wr_random:.1f}_draw{dr_self:.1f}.pth"
# # #             torch.save(policy.state_dict(), ckpt)
# # #             print(f"💾 Improved -> saved best_model.pth and {ckpt}")

# # #         # 你要的：>90 自动保存
# # #         if AUTO_SAVE and (wr_random >= SAVE_THRESHOLD) and (dr_self <= MAX_SELF_DRAW_FOR_SUBMIT):
# # #             torch.save(policy.state_dict(), "best_model_90plus.pth")
# # #             print("🏁 >=90% vs Random (and draw constraint) reached! Saved: best_model_90plus.pth")

# # #         # 持久在线 stats
# # #         with open("training_stats.json", "w", encoding="utf-8") as f:
# # #             json.dump(stats, f, ensure_ascii=False, indent=2)


# # # # 训练结束保存
# # # torch.save(policy.state_dict(), "best_model.pth")
# # # with open("training_stats.json", "w", encoding="utf-8") as f:
# # #     json.dump(stats, f, ensure_ascii=False, indent=2)

# # # # 画图（报告用）
# # # try:
# # #     plt.figure()
# # #     if len(stats["ep_return"]) > 0:
# # #         plt.plot(stats["ep_return"])
# # #         plt.title("Episode Return")
# # #         plt.xlabel("Episode")
# # #         plt.ylabel("Return")
# # #         plt.savefig("training_plot_return.png", dpi=200)
# # #     plt.close()

# # #     plt.figure()
# # #     if len(stats["loss"]) > 0:
# # #         # 只画最近一段更清晰
# # #         y = stats["loss"][-5000:] if len(stats["loss"]) > 5000 else stats["loss"]
# # #         plt.plot(y)
# # #         plt.title("Loss (last part)")
# # #         plt.xlabel("Update step")
# # #         plt.ylabel("Loss")
# # #         plt.savefig("training_plot_loss.png", dpi=200)
# # #     plt.close()

# # #     # eval plot
# # #     if len(stats["eval"]) > 0:
# # #         eps_list = [x["ep"] for x in stats["eval"]]
# # #         wr_list = [x["wr_random"] for x in stats["eval"]]
# # #         dr_list = [x["draw_self"] for x in stats["eval"]]

# # #         plt.figure()
# # #         plt.plot(eps_list, wr_list)
# # #         plt.title("WinRate vs Random")
# # #         plt.xlabel("Episode")
# # #         plt.ylabel("WinRate (%)")
# # #         plt.savefig("training_plot_wr_random.png", dpi=200)
# # #         plt.close()

# # #         plt.figure()
# # #         plt.plot(eps_list, dr_list)
# # #         plt.title("Self-Play Draw Rate")
# # #         plt.xlabel("Episode")
# # #         plt.ylabel("Draw (%)")
# # #         plt.savefig("training_plot_draw_self.png", dpi=200)
# # #         plt.close()

# # # except Exception as e:
# # #     print("Plotting failed:", e)

# # # print("✅ Done. Saved: best_model.pth, training_stats.json, training_plot_*.png")


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import json
# import os
# import matplotlib.pyplot as plt
# from collections import deque
# from minichess_env import MiniChessEnv


# # =========================
# # 1) 网络
# # =========================
# class MiniChessNet(nn.Module):
#     def __init__(self):
#         super(MiniChessNet, self).__init__()
#         self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 5 * 5, 256)
#         self.fc2 = nn.Linear(256, 625)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         return self.fc2(x)


# def board_to_tensor(board, device):
#     x = np.zeros((13, 5, 5), dtype=np.float32)
#     for r in range(5):
#         for c in range(5):
#             piece_idx = int(board[r, c])
#             x[piece_idx, r, c] = 1.0
#     return torch.tensor(x, device=device, dtype=torch.float32).unsqueeze(0)


# # =========================
# # 2) 超参数（你原版 + 少 draw 的关键项）
# # =========================
# BATCH_SIZE = 128
# GAMMA = 0.95
# EPS_START = 1.0
# EPS_END = 0.05
# EPS_DECAY = 10000
# LR = 1e-4
# TARGET_UPDATE = 10
# MEMORY_SIZE = 20000
# NUM_EPISODES = 30000

# # —— 少 draw：top-k 随机（推理/训练都用） —— #
# TOPK = 5
# TOPK_RANDOM_P_TRAIN = 0.05  # 训练时 exploitation: 5% 在 top-k 里随机
# TOPK_RANDOM_P_EVAL = 0.03   # 评估时更保守

# # —— 少 draw：重复局面检测 —— #
# REPEAT_LIMIT = 3            # 同一局面出现>=3次视为循环
# REPEAT_PENALTY = -8.0       # 循环惩罚（并 done=True）

# # —— 奖励缩放（稳定收敛） —— #
# WIN_REWARD = 10.0
# LOSE_REWARD = -10.0
# MATERIAL_SCALE = 0.5
# STEP_PENALTY = 0.05

# # —— 评估 —— #
# EVAL_EVERY = 1000
# EVAL_GAMES = 50
# EVAL_MAX_TURNS = 100

# # —— 自动保存阈值（vs random 胜率）—— #
# SAVE_WINRATE_THRESHOLD = 79.0


# # =========================
# # 3) 设备
# # =========================
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# print(f"Turbo Training Mode | Device: {device}")


# # =========================
# # 4) 模型与优化器
# # =========================
# policy_net = MiniChessNet().to(device)
# target_net = MiniChessNet().to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

# optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# criterion = nn.SmoothL1Loss()
# memory = deque(maxlen=MEMORY_SIZE)

# env = MiniChessEnv()
# if hasattr(env, "fig"):
#     plt.close(env.fig)


# # =========================
# # 5) 材质分（你的版本保留）
# # =========================
# PIECE_VALUES = {0: 0, 1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 100}

# def get_material_score(board, player):
#     score = 0
#     my_king = False
#     opp_king = False
#     my_k_id = 12 if player == 1 else 6
#     opp_k_id = 6 if player == 1 else 12

#     for r in range(5):
#         for c in range(5):
#             p = int(board[r, c])
#             if p == 0:
#                 continue
#             if p == my_k_id:
#                 my_king = True
#             if p == opp_k_id:
#                 opp_king = True
#             val = PIECE_VALUES[p if p <= 6 else p - 6]
#             if player == 1:
#                 score += val if p >= 7 else -val
#             else:
#                 score += val if p <= 6 else -val
#     return score, my_king, opp_king


# def state_key(board, current_player):
#     return board.tobytes() + bytes([1 if current_player == 1 else 0])


# # =========================
# # 6) 选动作：mask + argmax / top-k 随机
# # =========================
# def pick_action_from_q(q_vals, legal_moves, topk=5, p=0.05):
#     """
#     q_vals: (1,625) 已经 mask 过后的 q（非法为 -inf）
#     """
#     if len(legal_moves) <= 1:
#         return int(torch.argmax(q_vals, dim=1).item())

#     # 绝大多数时候走最优，少量概率走 top-k 随机打破循环
#     if random.random() > p:
#         return int(torch.argmax(q_vals, dim=1).item())

#     k = min(topk, len(legal_moves))
#     _, topk_idx = torch.topk(q_vals, k, dim=1)
#     j = int(torch.randint(0, k, (1,), device=q_vals.device).item())
#     return int(topk_idx[0, j].item())


# def select_action(policy_net, state_tensor, legal_moves, device, p_topk):
#     with torch.no_grad():
#         q = policy_net(state_tensor)
#         mask = torch.full((1, 625), -float('inf'), device=device, dtype=torch.float32)
#         mask[0, legal_moves] = 0.0
#         q_masked = q + mask
#         a = pick_action_from_q(q_masked, legal_moves, topk=TOPK, p=p_topk)
#         return a


# # =========================
# # 7) 评估：vs Random 输出 W/L/D（你现在缺这个）
# # =========================
# def eval_vs_random(agent_net, episodes=50, max_turns=100):
#     wins = losses = draws = 0
#     test_env = MiniChessEnv()
#     if hasattr(test_env, "fig"):
#         plt.close(test_env.fig)

#     for _ in range(episodes):
#         board, _ = test_env.reset()
#         player = 1
#         done = False

#         for _t in range(max_turns):
#             legal = test_env.get_legal_moves()
#             if not legal:
#                 # 当前 player 无合法步 => 当前 player 输
#                 if player == 1:
#                     losses += 1
#                 else:
#                     wins += 1
#                 done = True
#                 break

#             if player == 1:
#                 s = board_to_tensor(board, device)
#                 a = select_action(agent_net, s, legal, device, p_topk=TOPK_RANDOM_P_EVAL)
#                 if a not in legal:
#                     a = random.choice(legal)
#             else:
#                 a = random.choice(legal)

#             board, _, _, _, _ = test_env.step(a)
#             player *= -1

#             # 王被吃也算终局（环境不提供 done，只能我们自己判）
#             _, myk, oppk = get_material_score(board, player=1)
#             if not oppk:
#                 wins += 1
#                 done = True
#                 break
#             if not myk:
#                 losses += 1
#                 done = True
#                 break

#         if not done:
#             draws += 1

#     win_rate = wins / episodes * 100.0
#     draw_rate = draws / episodes * 100.0
#     return win_rate, draw_rate, wins, losses, draws


# # =========================
# # 8) 训练
# # =========================
# stats = {"rewards": [], "losses": [], "eval": []}
# steps_done = 0

# try:
#     for i_episode in range(NUM_EPISODES):
#         board, _ = env.reset()
#         env.move_log = []
#         state_tensor = board_to_tensor(board, device)
#         total_reward = 0.0
#         done = False

#         # 每局重复检测表
#         seen = {}

#         while not done:
#             # 我方回合开始时检查重复
#             k0 = state_key(board, env.current_player)
#             seen[k0] = seen.get(k0, 0) + 1
#             if seen[k0] >= REPEAT_LIMIT:
#                 # 直接判循环结束（强惩罚）
#                 total_reward += REPEAT_PENALTY
#                 done = True
#                 break

#             legal_moves = env.get_legal_moves()
#             if not legal_moves:
#                 # 我方无合法步 => 输
#                 total_reward += LOSE_REWARD
#                 done = True
#                 break

#             score_start, _, _ = get_material_score(board, player=1)

#             # epsilon
#             sample = random.random()
#             eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)
#             steps_done += 1

#             # 选动作
#             if sample > eps_threshold:
#                 action_idx = select_action(policy_net, state_tensor, legal_moves, device, p_topk=TOPK_RANDOM_P_TRAIN)
#             else:
#                 # 探索：偏向吃子（更快结束，少 draw）
#                 if random.random() < 0.6:
#                     cap = []
#                     for a in legal_moves:
#                         to_idx = a % 25
#                         tr, tc = divmod(to_idx, 5)
#                         if int(board[tr, tc]) != 0:
#                             cap.append(a)
#                     action_idx = random.choice(cap) if cap else random.choice(legal_moves)
#                 else:
#                     action_idx = random.choice(legal_moves)

#             # ===== 我方走一步 =====
#             board, _, _, _, _ = env.step(action_idx)

#             # 我方立即赢：对手无合法步 或 对手王没了
#             opp_legal = env.get_legal_moves()
#             _, _, opp_king_alive = get_material_score(board, player=1)
#             if (not opp_king_alive) or (not opp_legal):
#                 reward = WIN_REWARD
#                 memory.append((
#                     state_tensor,
#                     torch.tensor([[action_idx]], device=device, dtype=torch.int64),
#                     torch.tensor([reward], device=device, dtype=torch.float32),
#                     None,
#                     legal_moves
#                 ))
#                 total_reward += reward
#                 done = True
#                 break

#             # ===== 对手 random 走一步 =====
#             opp_action = random.choice(opp_legal)
#             board, _, _, _, _ = env.step(opp_action)

#             # ===== 结算 =====
#             score_end, my_king_alive, _ = get_material_score(board, player=1)
#             my_next_legal = env.get_legal_moves()

#             if (not my_king_alive) or (not my_next_legal):
#                 reward = LOSE_REWARD
#                 next_state_tensor = None
#                 done = True
#             else:
#                 # 材质变化 + 步数惩罚（稳定尺度）
#                 reward = (score_end - score_start) * MATERIAL_SCALE
#                 reward -= STEP_PENALTY
#                 next_state_tensor = board_to_tensor(board, device)

#             # ===== 对手走完后检查重复（最重要：防循环）=====
#             k1 = state_key(board, env.current_player)  # 此时轮到我方
#             seen[k1] = seen.get(k1, 0) + 1
#             if seen[k1] >= REPEAT_LIMIT:
#                 reward = REPEAT_PENALTY
#                 next_state_tensor = None
#                 done = True

#             # 存经验
#             memory.append((
#                 state_tensor,
#                 torch.tensor([[action_idx]], device=device, dtype=torch.int64),
#                 torch.tensor([reward], device=device, dtype=torch.float32),
#                 next_state_tensor,
#                 legal_moves
#             ))

#             state_tensor = next_state_tensor if next_state_tensor is not None else state_tensor
#             total_reward += float(reward)

#             # ===== 学习 =====
#             if len(memory) > BATCH_SIZE:
#                 transitions = random.sample(memory, BATCH_SIZE)
#                 batch_state, batch_action, batch_reward, batch_next_state, _ = zip(*transitions)

#                 batch_state = torch.cat(batch_state)
#                 batch_action = torch.cat(batch_action)
#                 batch_reward = torch.cat(batch_reward).float()

#                 current_q = policy_net(batch_state).gather(1, batch_action).squeeze(1)

#                 non_final_mask = torch.tensor(
#                     tuple(s is not None for s in batch_next_state),
#                     device=device,
#                     dtype=torch.bool
#                 )

#                 next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32)
#                 if non_final_mask.any():
#                     non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
#                     with torch.no_grad():
#                         next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

#                 expected_q = batch_reward + (GAMMA * next_state_values)
#                 loss = criterion(current_q, expected_q)

#                 optimizer.zero_grad()
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
#                 optimizer.step()
#                 stats["losses"].append(float(loss.item()))

#         if i_episode % TARGET_UPDATE == 0:
#             target_net.load_state_dict(policy_net.state_dict())

#         stats["rewards"].append(float(total_reward))

#         # ===== 评估与保存 =====
#         if i_episode % EVAL_EVERY == 0:
#             policy_net.eval()
#             wr, dr, w, l, d = eval_vs_random(policy_net, episodes=EVAL_GAMES, max_turns=EVAL_MAX_TURNS)
#             policy_net.train()

#             print(f"Ep {i_episode:5d} | eps {eps_threshold:.2f} | WR {wr:5.1f}% | DR {dr:5.1f}% | W/L/D={w}/{l}/{d}")

#             stats["eval"].append({
#                 "episode": int(i_episode),
#                 "eps": float(eps_threshold),
#                 "winrate_vs_random": float(wr),
#                 "drawrate_vs_random": float(dr),
#                 "W": int(w), "L": int(l), "D": int(d),
#             })

#             # 达标保存
#             if wr >= SAVE_WINRATE_THRESHOLD:
#                 torch.save(policy_net.state_dict(), "cuda2best_model.pth")
#                 print("Saved cuda2best_model.pth")

#             with open("training_stats.json", "w", encoding="utf-8") as f:
#                 json.dump(stats, f, ensure_ascii=False, indent=2)

# except KeyboardInterrupt:
#     print("Stopped.")

# finally:
#     print("Final Save...")
#     torch.save(policy_net.state_dict(), "best_model.pth")
#     with open("training_stats.json", "w", encoding="utf-8") as f:
#         json.dump(stats, f, ensure_ascii=False, indent=2)
#     plt.close("all")
#     print("Done.")







import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import matplotlib.pyplot as plt 
from collections import deque
from minichess_env import MiniChessEnv

class MiniChessNet(nn.Module):
    def __init__(self):
        super(MiniChessNet, self).__init__()

        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

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
            piece_idx = board[r, c]
            x[piece_idx, r, c] = 1.0
    return torch.tensor(x, device=device).unsqueeze(0)

BATCH_SIZE = 64     # 小 Batch 更新更频繁
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000    # 【关键】2000局内就把探索降到底，逼它快速学会
LR = 3e-4           # 【关键】高学习率，学得快
TARGET_UPDATE = 10
MEMORY_SIZE = 20000
NUM_EPISODES = 50000 

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Turbo Training Mode | Device: {device}")

policy_net = MiniChessNet().to(device)
target_net = MiniChessNet().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()
memory = deque(maxlen=MEMORY_SIZE)
env = MiniChessEnv()
plt.close(env.fig)

PIECE_VALUES = {0:0, 1:1, 2:3, 3:3, 4:5, 5:9, 6:100}

def get_material_score(board, player):
    score = 0
    my_king = False
    opp_king = False
    my_k_id = 12 if player == 1 else 6
    opp_k_id = 6 if player == 1 else 12

    for r in range(5):
        for c in range(5):
            p = board[r, c]
            if p == 0: continue
            if p == my_k_id: my_king = True
            if p == opp_k_id: opp_king = True
            val = PIECE_VALUES[p if p <=6 else p-6]
            if player == 1: 
                if p >= 7: score += val
                else: score -= val
            else: 
                if p <= 6: score += val
                else: score -= val
    return score, my_king, opp_king

def eval_win_rate(agent_net, episodes=20):
    wins = 0
    test_env = MiniChessEnv()
    plt.close(test_env.fig)
    
    for _ in range(episodes):
        board, _ = test_env.reset()
        done = False
        turns = 0
        player = 1
        while not done and turns < 50:
            legal = test_env.get_legal_moves()
            if not legal: break
            
            if player == 1: # AI
                with torch.no_grad():
                    t = board_to_tensor(board, device)
                    q = agent_net(t)
                    mask = torch.full((1, 625), -float('inf'), device=device)
                    mask[0, legal] = 0
                    action = (q + mask).max(1)[1].item()
            else: # Random Opponent
                action = random.choice(legal)
                
            board, _, _, _, _ = test_env.step(action)
            player *= -1
            turns += 1
            
            # Check win condition for AI (Player 1)
            # After step, player flipped. So if player is now -1, AI just moved.
            # Check if Opponent (-1) is dead
            opp_legal = test_env.get_legal_moves()
            _, _, opp_k = get_material_score(board, 1)
            
            if (not opp_k or not opp_legal) and player == -1:
                wins += 1
                done = True
                
    return (wins / episodes) * 100

stats = {"rewards": [], "losses": []}
steps_done = 0

try:
    for i_episode in range(NUM_EPISODES):
        board, _ = env.reset()
        env.move_log = [] 
        state_tensor = board_to_tensor(board, device)
        total_reward = 0
        player = 1 
        done = False
        
        while not done:
            # --- My Move ---
            legal_moves = env.get_legal_moves()
            if not legal_moves: break 

            score_start, _, _ = get_material_score(board, player)

            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if sample > eps_threshold:
                with torch.no_grad():
                    q = policy_net(state_tensor)
                    mask = torch.full((1, 625), -float('inf'), device=device)
                    mask[0, legal_moves] = 0 
                    action_idx = (q + mask).max(1)[1].item()
            else:
                action_idx = random.choice(legal_moves)

            board, _, _, _, _ = env.step(action_idx)
            
            opp_legal = env.get_legal_moves()
            _, _, opp_king_alive = get_material_score(board, player)
            
            if not opp_king_alive or not opp_legal:
                reward = 1400 
                memory.append((state_tensor, torch.tensor([[action_idx]], device=device), 
                               torch.tensor([reward], device=device), None, legal_moves))
                total_reward += reward
                done = True
                break
            
            # --- Opponent Move (Random) ---
            opp_action = random.choice(opp_legal)
            board, _, _, _, _ = env.step(opp_action)
            
            # --- Check Loss/Net Score ---
            score_end, my_king_alive, _ = get_material_score(board, player)
            my_next_legal = env.get_legal_moves()
            
            if not my_king_alive or not my_next_legal:
                reward = -800 
                next_state_tensor = None
                done = True
            else:
                reward = (score_end - score_start) * 30 
                
                reward -= 5 
                
                next_state_tensor = board_to_tensor(board, device)

            memory.append((state_tensor, torch.tensor([[action_idx]], device=device), 
                           torch.tensor([reward], device=device), next_state_tensor, legal_moves))
            
            state_tensor = next_state_tensor if next_state_tensor is not None else state_tensor
            total_reward += reward
            
            if len(memory) > BATCH_SIZE:
                transitions = random.sample(memory, BATCH_SIZE)
                batch_state, batch_action, batch_reward, batch_next_state, _ = zip(*transitions)
                
                batch_state = torch.cat(batch_state)
                batch_action = torch.cat(batch_action)
                batch_reward = torch.cat(batch_reward)
                
                current_q = policy_net(batch_state).gather(1, batch_action)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
                
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                if len(non_final_next_states) > 0:
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
                
                expected_q = batch_reward + (GAMMA * next_state_values)
                loss = criterion(current_q, expected_q.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                stats["losses"].append(loss.item())

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        stats["rewards"].append(total_reward)
        
        if i_episode % 1000 == 0:
            win_rate = eval_win_rate(policy_net)
            print(f"Ep {i_episode} | Eps: {eps_threshold:.2f} | Win Rate (vs Random): {win_rate}%")
            
            if win_rate >= 85:
                print("Saving:")
                torch.save(policy_net.state_dict(), "cuda2best_model.pth")

except KeyboardInterrupt:
    print("Stopped.")
finally:
    print("Final Save...")
    torch.save(policy_net.state_dict(), "cuda2best_model.pth")
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)
    plt.close('all')