import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from minichess_env import MiniChessEnv
import matplotlib.pyplot as plt


class MiniChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 625)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)



_SWAP = np.zeros(13, dtype=np.int8)
# 0 stays 0
# silver 1..6 <-> gold 7..12
for p in range(1, 7):
    _SWAP[p] = p + 6
    _SWAP[p + 6] = p

def _canon_board(board: np.ndarray, player: int) -> np.ndarray:

    if player == 1:
        return board
    # player == -1: rotate 180 degrees + swap sides encoding
    b = np.rot90(board, 2).copy()
    return _SWAP[b]

def _uncanon_action(action: int, player: int) -> int:
    if player == 1:
        return action
    # rotate 180: idx -> 24-idx
    from_idx = action // 25
    to_idx = action % 25
    from_idx = 24 - from_idx
    to_idx = 24 - to_idx
    return from_idx * 25 + to_idx

def _canon_action(action: int, player: int) -> int:
    if player == 1:
        return action
    from_idx = action // 25
    to_idx = action % 25
    from_idx = 24 - from_idx
    to_idx = 24 - to_idx
    return from_idx * 25 + to_idx

def _board_to_tensor(board: np.ndarray, device: torch.device) -> torch.Tensor:
    x = np.zeros((13, 5, 5), dtype=np.float32)
    # board values are 0..12
    for r in range(5):
        for c in range(5):
            x[int(board[r, c]), r, c] = 1.0
    return torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)

# piece values for capture bonus (reduce draw by forcing trades)
PIECE_VAL = {0:0, 1:1, 2:3, 3:3, 4:5, 5:9, 6:30}  # king value not “win” in env, but still high-ish


def _captured_value(board: np.ndarray, action: int) -> float:
    tr, tc = divmod(action % 25, 5)
    target = int(board[tr, tc])
    if target == 0:
        return 0.0
    base = target if target <= 6 else target - 6
    return float(PIECE_VAL.get(base, 0))


def _apply_on_copy(board: np.ndarray, action: int) -> np.ndarray:
    b = board.copy()
    from_idx = action // 25
    to_idx = action % 25
    fr, fc = divmod(from_idx, 5)
    tr, tc = divmod(to_idx, 5)
    b[tr, tc] = b[fr, fc]
    b[fr, fc] = 0
    return b


class Agent2:
    def __init__(self):
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        self.env = MiniChessEnv()
        plt.close(self.env.fig)

        self.model = MiniChessNet().to(self.device)
        self.model.eval()

        model_path = os.path.join(os.path.dirname(__file__), "cuda2best_model.pth")
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(sd)
            self.model.eval()
        else:
            self.model = None  # fallback random

        self.recent = deque(maxlen=24)
        self.eps_play = 0.03
        self.topk = 5

    def get_action(self, board, player):
        self.env.board = board.copy()
        self.env.current_player = player
        legal = self.env.get_legal_moves()
        if not legal:
            return 0

        if self.model is None:
            return random.choice(legal)

        b_c = _canon_board(board, player)
        legal_c = [_canon_action(a, player) for a in legal]

        if random.random() < self.eps_play:
            # still prefer captures a bit even when random
            legal_sorted = sorted(legal, key=lambda a: _captured_value(board, a), reverse=True)
            return legal_sorted[0] if random.random() < 0.6 else random.choice(legal)

        with torch.no_grad():
            s = _board_to_tensor(b_c, self.device)
            q = self.model(s)[0]  # (625,)

            mask = torch.full((625,), -1e9, device=self.device, dtype=torch.float32)
            mask[torch.tensor(legal_c, device=self.device)] = 0.0
            q = q + mask

            q_cpu = q.detach().cpu().numpy()

            for idx, a_orig in enumerate(legal):
                a_c = legal_c[idx]

                # capture bonus: encourage trades -> fewer 100-turn draws
                q_cpu[a_c] += 0.15 * _captured_value(board, a_orig)

                # repetition penalty: avoid cycles
                next_b = _apply_on_copy(board, a_orig)
                # after move, turn switches -> check repetition in opponent-to-move canonical
                next_hash = hash(_canon_board(next_b, -player).tobytes())
                if next_hash in self.recent:
                    q_cpu[a_c] -= 0.8

            topk = min(self.topk, len(legal_c))
            best_idx = np.argpartition(q_cpu, -topk)[-topk:]
            best_idx = best_idx[np.argsort(q_cpu[best_idx])[::-1]]
            chosen_c = int(random.choice(best_idx[:topk]))

        # update repetition memory with current state (canonical)
        self.recent.append(hash(b_c.tobytes()))
        chosen = _uncanon_action(chosen_c, player)

        # double safety
        if chosen not in legal:
            return random.choice(legal)
        return chosen
