
# Reinforcement Learning for Ms. Pac-Man

This repository features two different reinforcement learning implementations for the Ms. Pac-Man game using PyTorch. The first approach utilizes a Convolutional Neural Network (CNN) based policy network, while the second employs a Deep Q-Network (DQN) algorithm for value-based learning.

## CNN-Based Policy Network

# CNN Policy Network for Gym Environments

This repository contains a PyTorch implementation of a CNN-based policy network designed for OpenAI Gym environments, particularly demonstrated on the Ms. Pac-Man game.

## Features

- **PyTorch Neural Network:** Utilizes `torch.nn` module to create a CNN (Convolutional Neural Network) architecture for policy learning.
  
- **Preprocessing:** Includes image preprocessing for Gym environments with `torchvision.transforms` and `PIL` (Python Imaging Library).

- **Policy Gradient Method:** Implements a policy gradient method for direct policy optimization without value function estimation.

## Implementation

- **CNN Architecture:** The `CNNGymPolicy` class defines a CNN with three convolutional layers and two fully connected layers, tailored to process Gym environment states.

- **Optimization:** Employs `torch.optim.Adam` for policy network optimization with a learning rate of 0.0001.

- **Training Loop:** Runs a training loop for 1000 episodes, using policy gradient methods to update the network weights based on the computed returns.

- **Reward Tracking:** Calculates cumulative average rewards and rolling average rewards over the last 10 episodes to monitor the agent's performance.

- **Visualization:** Visualizes the cumulative average and rolling average rewards over episodes using `seaborn` and `matplotlib`.

## Requirements

- Python 3.6 or above
- PyTorch
- OpenAI Gym
- torchvision
- PIL
- numpy
- seaborn
- matplotlib
- pandas

To run the code, make sure you have all the required libraries installed. You can clone the repository and execute the Python script to start training the agent in the Ms. Pac-Man environment.

## Sample Output

The training loop prints out the episode number, episode reward, cumulative average, and average of the last 10 episodes after each episode. It also produces plots showing the Episodes vs. Cumulative Average and the Average Reward of the Last 10 Episodes Over Time.

## Acknowledgements

This implementation is based on the REINFORCE algorithm, leveraging the simplicity and power of CNNs to learn policies in complex environments like Ms. Pac-Man.

## Deep Q-Network (DQN)

# DQN for Ms. Pac-Man in PyTorch

This repository contains the extracted Python code from a Jupyter notebook that implements the Deep Q-Network (DQN) algorithm for the Ms. Pac-Man game using PyTorch.

## Overview

The code includes the following components:

- **DQN Network:** A deep neural network architecture designed to approximate the Q-values for the Ms. Pac-Man game state-action pairs.
- **Environment Wrappers:** Preprocessing steps applied to the environment's observations for improved learning efficiency.
- **Training Loop:** The main loop where the DQN agent learns from interactions with the environment by updating the Q-network.
- **Hyperparameters:** Settings for learning rate, discount factor, batch size, and other parameters crucial for training the DQN agent.
- **Visualizations:** Code for plotting the performance metrics such as the rolling average of rewards.

## Requirements

- Python 3
- PyTorch
- OpenAI Gym
- Matplotlib
- Numpy

Make sure to install all necessary dependencies to run the Python script successfully.

## Usage

Run the script in a Python environment where all dependencies have been installed. The script will initialize the environment, train the DQN agent, and output the performance plots.

## Additional Notes

- The script is extracted from a Jupyter notebook, and it's recommended to understand the context from the original notebook.
- The agent's performance and training duration can vary based on the machine's computation power.

## Requirements

- Python 3.6 or above
- PyTorch
- OpenAI Gym
- torchvision
- PIL
- numpy
- seaborn (for CNN Policy Network)
- matplotlib
- pandas (for CNN Policy Network)

## Getting Started

To get started with these implementations:
1. Clone the repository.
2. Install the required dependencies listed above.
3. Run the corresponding Python scripts to train the agents.

## Acknowledgements

These implementations are based on well-known algorithms in the reinforcement learning community, adapted for the Ms. Pac-Man environment to demonstrate the effectiveness of CNNs and DQNs in game-playing AI.


