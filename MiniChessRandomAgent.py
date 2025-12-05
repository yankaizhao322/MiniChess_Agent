#Random Agent
class Agent:
    def __init__(self):
        # Optional: load model
        pass

    def get_action(self, board, player):
        # board: 5x5 numpy array
        # player: 1 (gold) or -1 (silver)
        # Must return a legal action (int in [0, 624])
        import random
        return random.choice(self.get_legal_moves(board, player))

    def get_legal_moves(self, board, player):
        from minichess_env import MiniChessEnv
        env = MiniChessEnv()
        env.board = board.copy()
        env.current_player = player
        return env.get_legal_moves()
