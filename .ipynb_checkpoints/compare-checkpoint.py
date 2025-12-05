import torch
import numpy as np
from minichess_env import MiniChessEnv

# 1. 导入你的 Agent 类
# 注意：你的截图里文件名是 my_agent2.py，类名是 Agent2
from my_agent2 import Agent2 

# ================= 配置区域 =================
# 在这里填入你想对比的两个模型路径
MODEL_PATH_B = '100_best_model.pth'  # 扮演 Gold (先手)
MODEL_PATH_A = 'submit_best_model.pth'  # 扮演 Silver (后手)
# ===========================================

def get_agent_with_weights(pth_path):
    """实例化 Agent 并强制加载指定的权重文件"""
    print(f"正在初始化 Agent 并加载: {pth_path} ...")
    
    agent = Agent2()
    
    # 2. 强制覆盖权重
    # map_location='cuda:0' 或 'cpu' 取决于你的环境，这里设为自动
    device = agent.device 
    state_dict = torch.load(pth_path, map_location=device)
    
    # 3. 载入新权重
    agent.model.load_state_dict(state_dict)
    agent.model.eval() # 确保进入评估模式
    
    return agent

def play_match(agent_gold, agent_silver):
    env = MiniChessEnv()
    obs, _ = env.reset()
    
    # 1 代表 Gold (先手), -1 代表 Silver (后手)
    players = {1: agent_gold, -1: agent_silver}
    current_player = 1 # Gold 先手
    max_turns = 1000
    
    print(f"\n🔥 对局开始: {MODEL_PATH_A} (Gold) VS {MODEL_PATH_B} (Silver)")
    
    for turn in range(max_turns):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            winner = -current_player
            print(f"结果: {'Gold' if current_player==1 else 'Silver'} 无路可走，判负。")
            return winner

        # 获取当前玩家的 Agent
        active_agent = players[current_player]
        
        # 获取动作
        # 注意：这里传入 env.board.copy() 以防 Agent 修改原始棋盘
        action = active_agent.get_action(env.board.copy(), current_player)
        
        if action not in legal_moves:
            print(f"非法动作! 玩家 {current_player} 尝试了 {action}")
            return -current_player
            
        env.step(action)
        
        # 可选：打印每一步（如果嫌太长可以注释掉）
        # env.render() 
        
        current_player *= -1
        
    print("结果: 平局 (达到最大回合数)")
    return 0

if __name__ == "__main__":
    # 1. 加载两个不同权重的 Agent
    try:
        player_a = get_agent_with_weights(MODEL_PATH_A)
        player_b = get_agent_with_weights(MODEL_PATH_B)
        
        # 2. 开始对战
        winner = play_match(player_a, player_b)
        
        print("-" * 30)
        if winner == 1:
            print(f"🏆 最终获胜: Gold ({MODEL_PATH_A})")
        elif winner == -1:
            print(f"🏆 最终获胜: Silver ({MODEL_PATH_B})")
        else:
            print("🤝 最终结果: 平局")
            
        print("-" * 30)

        
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件，请检查路径名称。\n{e}")
    except Exception as e:
        print(f"❌ 发生错误: {e}")