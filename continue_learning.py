import gym
import envs
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import PPO2



env = gym.make('CarRacing-v0')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #env.render()

env = make_atari_env('CarRacing-v1',num_env=1, seed=0)

# change env
model.set_env(env)
model.learn(total_timesteps=1000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
