# Deep Q-Learning Agent for MiniChess ♟️

A robust Reinforcement Learning (RL) agent capable of playing **5x5 MiniChess** at a high strategic level. This project implements a **Deep Q-Network (DQN)** architecture augmented with **Convolutional Neural Networks (CNN)** and utilizes a **Hyper-Aggressive Reward Shaping** strategy to solve the common problem of passive play (draws) in chess-like environments.

Developed for **ECE 340/440: Introduction to Machine Learning / Deep Learning** at Lehigh University.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Performance](#performance)
- [Project Structure](#project-structure)

## 🛠 Project Overview
MiniChess is played on a $5 \times 5$ board with simplified rules but complex tactical depth. The objective is to **checkmate (capture) the opponent's King**.

The goal of this project was to train an agent from scratch without human knowledge (tabula rasa) to:
1.  Always execute **legal moves**.
2.  Demonstrate **strategic resolution** (minimize draws).
3.  Defeat random baselines and compete in a round-robin tournament.

## ✨ Key Features
* **DQN with Experience Replay**: Breaks correlations in training data for stable learning.
* **Convolutional State Processing**: A custom 2-layer CNN extracts spatial features from the $13 \times 5 \times 5$ board tensor representation.
* **Action Masking**: A safety layer in the network output ensures the agent assigns $-\infty$ Q-values to illegal moves, guaranteeing 100% validity.
* **Hyper-Aggressive Policy**: Solved the "stalling/draw" problem by implementing a severe step penalty and massive win rewards.
* **Canonicalization**: The agent internally rotates the board to always play from a "Gold" perspective, doubling data efficiency.

## 💻 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/minichess-dqn.git](https://github.com/your-username/minichess-dqn.git)
    cd minichess-dqn
    ```

2.  Install dependencies:
    ```bash
    pip install numpy torch matplotlib gymnasium
    ```

## 🚀 Usage

### 1. Run the Agent (vs Random)
To see the agent play against a baseline Random Agent:
```bash
python verify.py
