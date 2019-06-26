import gym
import envs
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C

env = gym.make('CarRacing-v1')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model._load_from_file('D://Users//Han//Workspace//gym_learn//PPO2_50K')
# model.learn(total_timesteps=5000, log_interval=100)
# model._save_to_file('D://Users//Han//Workspace//gym_learn//PPO2_50K')

# model._load_from_file('D://Users//Han//Workspace//gym_learn//PPO2_50K')

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
