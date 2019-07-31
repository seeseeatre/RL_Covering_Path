import os

import gym
import envs
import numpy as np
import matplotlib.pyplot as plt
import random

#from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines import DDPG
#from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines import PPO2, A2C
from stable_baselines import TRPO


best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

# Create log dir
log_dir = "D:/Users/Han/Workspace/gym_learn/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make('CarRacing-v1')


env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# Add some param noise for exploration
#param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization
#model = DDPG(LnMlpPolicy, env, param_noise=param_noise, verbose=0)
# Train the agent
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
# env = Monitor(env, log_dir, allow_early_resets=True)

# model = PPO2(CnnLstmPolicy, env, verbose=1, nminibatches=1, tensorboard_log="./test_tensorboard/")
# n_steps=10, gamma=0.75, 
model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log="./test_tensorboard/")
model._load_from_file('D://Users//Han//Workspace//gym_learn//model//TRPO_MlpPolicy_PIX_SMALL_EMPTY_TTD.pkl')
model.learn(total_timesteps=int(1e5), callback=callback)
model._save_to_file('D://Users//Han//Workspace//gym_learn//model//TRPO_MlpPolicy_PIX_SMALL_EMPTY_stack.pkl')
plot_results(log_dir)

obs = env.reset()
reward_sum = 0.0
action_history = np.zeros(6)
action_sum = np.zeros(6)
ep_length = 0
ep_length_record = []
for i in range(10000):
	action = [random.randint(0,5)]
	action_history[action]+=1
	action_sum[action]+=1
	obs, rewards, dones, info = env.step(action)
	reward_sum += rewards

	ep_length += 1

	# if i % 100 == 0:
	# 	import matplotlib.pyplot as plt
	# 	plt.imshow(obs[0]*255)
	# 	plt.savefig("test.jpeg")

	env.render()
	if dones:
		# import matplotlib.pyplot as plt
		# plt.imshow(obs[0]*255)
		# plt.savefig("test.jpeg")
		
		print("reward: ", reward_sum)
		reward_sum = 0.0

		print("actions: ", action_history/ep_length)
		action_history = np.zeros(6)

		print("Episode length: ", ep_length)
		ep_length_record.append(ep_length)
		ep_length = 0

		obs = env.reset()
if ep_length >0:
  ep_length_record.append(ep_length)

print("all actions: ", action_sum/len(ep_length_record))
ep_length_avg = sum(ep_length_record) / len(ep_length_record)
print("Average Episode Length: ", ep_length_avg)