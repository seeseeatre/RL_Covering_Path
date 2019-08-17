import gym
import envs
import numpy as np
from stable_baselines import DQN

from stable_baselines.gail import generate_expert_traj

env = gym.make("CarRacing-v1")

model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
generate_expert_traj(model, 'expert_cartpole', env,n_timesteps=int(1e5), n_episodes=10, image_folder='recorded_images')
