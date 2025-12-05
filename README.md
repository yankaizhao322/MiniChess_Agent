# Deep Q-Learning Agent for MiniChess

A robust Reinforcement Learning (RL) agent capable of playing **5x5 MiniChess** at a high strategic level. This project implements a **Deep Q-Network (DQN)** architecture augmented with **Convolutional Neural Networks (CNN)** and utilizes a **Hyper-Aggressive Reward Shaping** strategy to solve the common problem of passive play (draws) in chess-like environments.

Developed for **ECE 340/440: Introduction to Machine Learning / Deep Learning** at Lehigh University.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Performance](#performance)
- [Project Structure](#project-structure)

## Project Overview
MiniChess is played on a $5 \times 5$ board with simplified rules but complex tactical depth. The objective is to **checkmate (capture) the opponent's King**.

The goal of this project was to train an agent from scratch without human knowledge (tabula rasa) to:
1.  Always execute **legal moves**.
2.  Demonstrate **strategic resolution** (minimize draws).
3.  Defeat random baselines and compete in a round-robin tournament.

## Key Features
* **DQN with Experience Replay**: Breaks correlations in training data for stable learning.
* **Convolutional State Processing**: A custom 2-layer CNN extracts spatial features from the $13 \times 5 \times 5$ board tensor representation.
* **Action Masking**: A safety layer in the network output ensures the agent assigns $-\infty$ Q-values to illegal moves, guaranteeing 100% validity.
* **Hyper-Aggressive Policy**: Solved the "stalling/draw" problem by implementing a severe step penalty and massive win rewards.
* **Canonicalization**: The agent internally rotates the board to always play from a "Gold" perspective, doubling data efficiency.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/minichess-dqn.git](https://github.com/your-username/minichess-dqn.git)
    cd minichess-dqn
    ```

2.  Install dependencies:
    ```bash
    pip install numpy torch matplotlib gymnasium
    ```

## Usage

### 1. Run the Agent (vs Random)

To see the agent play against a baseline Random Agent:
```bash
python verify.py


### Train from Scratch
To retrain the model using the hyper-aggressive configuration:

```Bash

python train_best.py
```

### Load the Agent

The Agent class in my_agent.py automatically loads the pre-trained weights from best_model.pth.

```Python
from my_agent import Agent
agent = Agent()
action = agent.get_action(board, player)
```

### Training Strategy
To overcome the issue where agents learn to "survive" rather than "win" (leading to 100-turn draws), this project used a Magnitude Scaling reward function:

1.   Win Reward: +100,000 (Forces the agent to prioritize checkmate above all else).
    
2.   Step Penalty: -20 per turn (Forces the agent to win quickly).
    
3.   Loss Penalty: -20,000.
    
4.   Gamma ($\gamma$): 0.90 (Short-sightedness encouraged to prevent hesitation).

### Performance
Self-Play Evaluation
In a 10-game self-play experiment, the agent demonstrated exceptional decisiveness:

-  Decisive Games: 90% (Win/Loss)

-  Draws: 10%

-  Conclusion: The agent successfully avoids passive loops and actively hunts for the King.

Benchmark vs. Random

-  Win Rate: ~67%

- Note: The 19% loss rate is a calculated trade-off of the hyper-aggressive policy, which occasionally takes tactical risks to speed up the game.

## Project Structure
- my_agent.py: The main submission file containing the Agent class and CNN model.

- best_model.pth: Pre-trained model weights (Loaded by my_agent.py).

- minichess_env.py: The game environment provided by the course.

- train_best.py: The training script used to generate the model.

- verify.py: Script to test win rates against a random opponent.

- Report.pdf: Final project report.
