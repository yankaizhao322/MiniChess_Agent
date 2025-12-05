import matplotlib
# 【关键修复】必须在导入 pyplot 之前设置，强制使用非交互式后端
matplotlib.use('Agg') 

from match_runner import play_game
import my_agent
import MiniChessRandomAgent
import matplotlib.pyplot as plt

# 初始化
my_bot = my_agent.Agent()
random_bot = MiniChessRandomAgent.Agent()

wins = 0
losses = 0
draws = 0
num_games = 100

print(f"Running {num_games} games verification (Crash-Proof Mode)...")

try:
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end="\r")
        
        # verbose=False 不打印每一步，加快速度
        # 即使 verbose=False，环境内部还是可能会创建 figure，所以我们需要 plt.close
        winner, reason = play_game(my_bot, random_bot, verbose=False)
        
        if winner == 1: wins += 1
        elif winner == -1: losses += 1
        else: draws += 1
        
        # 【关键修复】每局必关
        plt.close('all') 

except Exception as e:
    print(f"\nError occurred: {e}")

print(f"\nFinal Results:")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Draws: {draws}")
print(f"Win Rate: {wins/num_games*100}%")