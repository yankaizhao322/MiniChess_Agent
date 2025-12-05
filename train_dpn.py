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

BATCH_SIZE = 128     # 小 Batch 更新更频繁
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000    # 【关键】2000局内就把探索降到底，逼它快速学会
LR = 3e-4           # 【关键】高学习率，学得快
TARGET_UPDATE = 10
MEMORY_SIZE = 20000
NUM_EPISODES = 50000 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                reward = 2000 
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
                reward = -1000 
                next_state_tensor = None
                done = True
            else:
                reward = (score_end - score_start) * 40 
                
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
                torch.save(policy_net.state_dict(), "best_model.pth")

except KeyboardInterrupt:
    print("Stopped.")
finally:
    print("Final Save...")
    torch.save(policy_net.state_dict(), "best_model.pth")
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)
    plt.close('all')
