from match_runner import play_game
import my_agent
import matplotlib.pyplot as plt

# 加载两个完全一样的你的 Agent
agent1 = my_agent.Agent() # 执金
agent2 = my_agent.Agent() # 执银 (也是你自己)

results = {"Gold": 0, "Silver": 0, "Draw": 0}
games_log = [] # 记录每局是谁赢了，用于写 Case Study

print(">>> 开始 10 局自我对弈实验 (Self-Play Experiment)...")

for i in range(10):
    print(f"正在进行第 {i+1} 局...", end="")
    
    # 这一步极其关键：verbose=False 不画图，跑得快；
    # 如果你想截图写 Case Study，把其中一局改成 True
    winner, reason = play_game(agent1, agent2, verbose=False)
    plt.close('all') # 清理内存
    
    if winner == 1: 
        results["Gold"] += 1
        res_str = "Gold Wins"
    elif winner == -1: 
        results["Silver"] += 1
        res_str = "Silver Wins"
    else: 
        results["Draw"] += 1
        res_str = "Draw"
        
    print(f" 结果: {res_str} | 原因: {reason}")
    games_log.append(f"Game {i+1}: {res_str} ({reason})")

print("\n=== 实验结果统计 ===")
print(f"总局数: 10")
print(f"金方胜 (先手): {results['Gold']}")
print(f"银方胜 (后手): {results['Silver']}")
print(f"平局: {results['Draw']}")
print("==================")

if results['Draw'] <= 2:
    print("✅ 完美！平局很少，符合 Strategic Resolution 要求。")
else:
    print("⚠️ 警告：平局稍多，报告里可能需要解释原因（如防守太强）。")