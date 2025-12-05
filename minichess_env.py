

import gymnasium as gym
from gymnasium import spaces
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import matplotlib.image as mpimg
import os

class MiniChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 5x5 board, 13 states per square (empty + 12 pieces)
        self.observation_space = spaces.Box(low=0, high=12, shape=(5, 5), dtype=np.int8)

        # 25 from-squares x 25 to-squares = 625 possible moves
        self.action_space = spaces.Discrete(625)

        self.board = np.zeros((5, 5), dtype=np.int8)
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.set_xlim(0, 7)  # Extra space for move log
        self.ax.set_ylim(0, 5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        self.move_log = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Setup initial board state
        self.board = np.zeros((5, 5), dtype=np.int8)

        # Place black pieces (1–6)
        self.board[0] = [4, 2, 3, 5, 6]  # r n b q k
        self.board[1] = [1] * 5  # pawns

        # Place white pieces (7–12)
        self.board[4] = [10, 8, 9, 11, 12]  # R N B Q K
        self.board[3] = [7] * 5  # pawns

        self.current_player = 1  # 1=white, -1=black
        self.done = False
        return self.board, {}


    def render(self):
        clear_output(wait=True)
#        self.ax.clear()
        fig, ax = plt.subplots(figsize=(7,5))
        # Draw 5x5 board
        square_colors = ['#EEEED2', '#769656']
        ax.set_xlim(0,7)
        ax.set_ylim(0,5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        for r in range(5):
            for c in range(5):
                color = square_colors[(r + c) % 2]
                #self.ax.add_patch(plt.Rectangle((c, 4 - r), 1, 1, color=color))
                ax.add_patch(plt.Rectangle((c,4-r),1,1,color=color))

        # Highlight last move
        if self.move_log:
            _, _, to_row, to_col = self.move_log[-1]
            ax.add_patch(plt.Rectangle((to_col,4-to_row),1,1,color='yellow',alpha=0.4))
#            self.ax.add_patch(plt.Rectangle(
 #               (to_col, 4 - to_row), 1, 1,
  #              color='yellow', alpha=0.4
   #         ))

        # Map piece values to image files
        piece_map = {
            1: 'p_sil_tr.png', 2: 'n_sil_tr.png', 3: 'b_sil_tr.png', 4: 'r_sil_tr.png', 5: 'q_sil_tr.png', 6: 'k_sil_tr.png',
            7: 'p_gld_tr.png', 8: 'n_gld_tr.png', 9: 'b_gld_tr.png', 10: 'r_gld_tr.png', 11: 'q_gld_tr.png', 12: 'k_gld_tr.png'
        }

        # Draw pieces
        for r in range(5):
            for c in range(5):
                piece = self.board[r, c]
                if piece != 0:
                    file = piece_map[piece]
                    path = os.path.join("minichess_assets", file)
                    img = mpimg.imread(path)
#                    print("Rendering piece:", piece,"->", path)
                    ax.imshow(img, extent=(c, c + 1, 4 - r, 5 - r), zorder=10)

        # Move log text
        log_x = 5.5
        for i, move in enumerate(reversed(self.move_log[-5:])):
            fr, fc, tr, tc = move
            move_str = f"{5 - fr}{chr(fc + 97)} → {5 - tr}{chr(tc + 97)}"
            ax.text(log_x, 4.8 - i, move_str, fontsize=10, va='top')

#        self.fig.canvas.draw()
#        plt.show(block=False)
  #      self.fig.canvas.flush_events()
        plt.tight_layout()
        display(fig)
        plt.close(fig)
    def get_legal_moves(self):
        legal_moves = []
        for from_row in range(5):
            for from_col in range(5):
                piece = self.board[from_row, from_col]
                if piece == 0:
                    continue

                is_white = piece >= 7
                if (self.current_player == 1 and not is_white) or (self.current_player == -1 and is_white):
                    continue  # skip opponent pieces

                moves = self.get_piece_moves(piece, from_row, from_col)
                for to_row, to_col in moves:
                    # Validate destination: can't capture own piece
                    target = self.board[to_row, to_col]
                    if target == 0 or (is_white != (target >= 7)):
                        from_idx = from_row * 5 + from_col
                        to_idx = to_row * 5 + to_col
                        legal_moves.append(from_idx * 25 + to_idx)
        return legal_moves










    def get_piece_moves(self, piece, r, c):
        moves = []
        is_white = piece >= 7
        direction = -1 if is_white else 1
        board = self.board

        def on_board(x, y):
            return 0 <= x < 5 and 0 <= y < 5

        def is_enemy(x, y):
            target = board[x, y]
            return target != 0 and (is_white != (target >= 7))

        def is_empty(x, y):
            return board[x, y] == 0

        if piece in [1, 7]:  # Pawn
            fr = r + direction
            # forward
            if on_board(fr, c) and is_empty(fr, c):
                moves.append((fr, c))
            # captures
            for dc in [-1, 1]:
                fc = c + dc
                if on_board(fr, fc) and is_enemy(fr, fc):
                    moves.append((fr, fc))

        elif piece in [2, 8]:  # Knight
            for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                           (1, -2), (1, 2), (2, -1), (2, 1)]:
                x, y = r + dr, c + dc
                if on_board(x, y) and (is_empty(x, y) or is_enemy(x, y)):
                    moves.append((x, y))

        elif piece in [3, 9]:  # Bishop
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                for i in range(1, 5):
                    x, y = r + dr*i, c + dc*i
                    if not on_board(x, y):
                        break
                    if is_empty(x, y):
                        moves.append((x, y))
                    elif is_enemy(x, y):
                        moves.append((x, y))
                        break
                    else:
                        break

        elif piece in [4, 10]:  # Rook
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                for i in range(1, 5):
                    x, y = r + dr*i, c + dc*i
                    if not on_board(x, y):
                        break
                    if is_empty(x, y):
                        moves.append((x, y))
                    elif is_enemy(x, y):
                        moves.append((x, y))
                        break
                    else:
                        break

        elif piece in [5, 11]:  # Queen = rook + bishop
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                           (-1, 0), (1, 0), (0, -1), (0, 1)]:
                for i in range(1, 5):
                    x, y = r + dr*i, c + dc*i
                    if not on_board(x, y):
                        break
                    if is_empty(x, y):
                        moves.append((x, y))
                    elif is_enemy(x, y):
                        moves.append((x, y))
                        break
                    else:
                        break

        elif piece in [6, 12]:  # King
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    x, y = r + dr, c + dc
                    if on_board(x, y) and (is_empty(x, y) or is_enemy(x, y)):
                        moves.append((x, y))

        return moves












    def step(self, action):
        legal = self.get_legal_moves()
        if action not in legal:
            return self.board, -1, False, False, {"error": "Illegal move"}

        # Decode move
        from_idx = action // 25
        to_idx = action % 25
        fr, fc = divmod(from_idx, 5)
        tr, tc = divmod(to_idx, 5)

        piece = self.board[fr, fc]
        self.board[tr, tc] = piece
        self.board[fr, fc] = 0
        self.move_log.append((fr, fc, tr, tc))

        self.current_player *= -1
        reward = 0  # or +1 if checkmate later
        done = False
        info={}
        return self.board, reward, done, False, info
