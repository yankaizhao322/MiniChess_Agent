import importlib
from minichess_env import MiniChessEnv

def load_agent(path):
    module = importlib.import_module(path.replace('.py', ''))
    return module.Agent()

def play_game(agent1, agent2, verbose=True):
    env = MiniChessEnv()
    obs, _ = env.reset()
    agents = {1: agent1, -1: agent2}
    player = 1
    max_turns = 100

    for turn in range(max_turns):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            winner = -player
            reason = f"No legal moves for {'Gold' if player == 1 else 'Silver'}"
            break

        try:
            action = agents[player].get_action(env.board.copy(), player)
        except Exception as e:
            winner = -player
            reason = f"{'Gold' if player == 1 else 'Silver'} crashed: {str(e)}"
            break

        if action not in legal_moves:
            winner = -player
            reason = f"Illegal move by {'Gold' if player == 1 else 'Silver'}"
            break

        env.step(action)
        if verbose:
            env.render()

        player *= -1

    else:
        winner = 0
        reason = "Draw: Max turns reached"

    print(f"Game over: Winner = {'Gold' if winner == 1 else 'Silver' if winner == -1 else 'Draw'}")
    print(f"Reason: {reason}")
    return winner, reason
