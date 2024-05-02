# AI Kung Fu for Atari Games

This repository provides a comprehensive guide and implementation details for a reinforcement learning agent based on the Asynchronous Advantage Actor-Critic (A3C) algorithm. This specific setup is tailored for playing Atari games, employing a convolutional neural network for decision making and including a robust preprocessing pipeline for image data.

## Table of Contents
- [Installation](#installation)
- [Overview](#overview)
- [Architecture](#architecture)
- [Usage](#usage)
- [Training the Agent](#training-the-agent)
- [Evaluation](#evaluation)
- [Visualization](#visualization)

## Installation

Before running the code, ensure that the following libraries are installed:
- `cv2` (OpenCV for image processing)
- `numpy`
- `torch` (PyTorch)
- `gymnasium` (OpenAI Gym)

These can be installed using pip:

```bash
pip install opencv-python numpy torch
pip install gymnasium
pip install gymnasium[atari, accept-rom-license]
apt-get install -y swig
pip install gymnasium[box2d]
```

## Overview

The code is structured to simulate an Atari game environment using `gymnasium`, preprocess game frames for neural network compatibility, define an A3C neural network architecture, and train multiple agents asynchronously.

## Architecture

### Neural Network

The neural network is defined in the `neural_network` class. It includes:
- Three convolutional layers for feature extraction from frames.
- Flattened output feeds into two separate fully connected layers:
  - `fc2a` for action prediction.
  - `fc2v` for state value estimation.
  
This network outputs both the action probabilities and the estimated value of the state, essential components of the A3C algorithm.

### Preprocessing

The `PreprocessAtari` class is an observation wrapper that:
- Converts frames to grayscale (if `color=False`).
- Resizes frames to a specified dimension.
- Normalizes pixel values.
- Stacks a specified number of frames together to provide temporal context to the network.

### Agent and Environment Interaction

- `Agent` class: Handles the action decision process and learning process of the neural network.
- `EnvBatch` class: Manages a batch of environments to facilitate asynchronous training of multiple agents.

## Usage

### Initializing the Environment

```python
env = make_env()
```

This function initializes the game environment with preprocessing settings applied.

### Training the Agent

Training is handled by iterating over episodes and updating the network's weights based on the rewards and the computed gradients.

```python
env_batch = EnvBatch(number_actions)
batch_states = env_batch.reset()
# Training loop here...
```

## Training the Agent

The training loop runs for a defined number of iterations, using the `tqdm` library to provide a progress bar. The agent interacts with the environment, and after each action, it receives feedback which is used to update the model's weights.

## Evaluation

After training, the agent's performance can be evaluated:

```python
average_rewards = evaluate(agent, env, n_episodes=10)
print("Average agent reward:", np.mean(average_rewards))
```

## Visualization

The `show_video_of_model` function is used to record a video of the agent playing the game, which can be displayed using the `show_video` function:

```python
show_video_of_model(agent, env)
show_video()
```

This sequence generates and displays a video, providing a visual feedback mechanism to assess the agent's performance.
