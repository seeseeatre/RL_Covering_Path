import os
import argparse
import gym
import envs
import numpy as np
import matplotlib.pyplot as plt

#from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
#from stable_baselines import DDPG
#from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines import PPO2, A2C, DQN, TRPO
plt.ion()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an RL agent for map coverage')
    parser.add_argument('--rl', type=str, nargs=1, default='ALL',
                        help="""The RL algorithm to use. Possible values are "HER", "DDPG", "GAIL", "SAC", "TRPO",
                        '"A2C", "PPO2" or "ALL". Default is ALL""")
    parser.add_argument('--policy', type=str, nargs=1, default='ALL',
                        help="""The policy algorithm to use. Possible values are "MlpLstmPolicy", "MlpPolicy", 
                        "MlpLnLstmPolicy", "MlpLstmPolicy", "CnnPolicy", "CnnLstmPolicy", "CnnLnLstmPolicy",
                         or "ALL". Default is ALL""")
    args = parser.parse_args()

    policy_types = ["MlpPolicy","DQMlpPolicy"]
    if args.policy in policy_types:
        policy_types = [args.policy]
    elif not args.policy == "ALL":
        raise Exception(f"Unrecognised policy type {args.policy}")
    RL_types = ["DQN", "TRPO", "A2C", "PPO2"]
    #RL_types = ["TRPO", "A2C"]
    if args.rl in RL_types:
        RL_types = [args.rl]
    elif not args.rl == "ALL":
        raise Exception(f"Unrecognised RL type {args.rl}")

    
    for RL_algorithm in RL_types:
        if RL_algorithm == "DQN":
            Policy_type = "DQMlpPolicy"
        else:
            Policy_type = "MlpPolicy"
        nattempts = 1
        for attempt in range(1, nattempts+1):
            print(f'{RL_algorithm}_{Policy_type}_GRD{attempt}')
            tf_log_file = f"./test_tensorboard/grid/stage3_gg/{Policy_type}/"
            model_file = f'{tf_log_file}/{RL_algorithm}_grid_stage3_gg.pkl'


            if os.path.isfile(f'{model_file}_grid_stage3_gg.pkl'):
                print("skipping previously completed test")
                continue
            try:
                # Create log dir
                # Create and wrap the environment
                env = gym.make('CarRacing-v2')
                log_dir = f"{model_file}_log/"
                os.makedirs(log_dir, exist_ok=True)
                env = Monitor(env, log_dir, allow_early_resets=True)
                env = DummyVecEnv([lambda: env])

                model = eval(RL_algorithm)(eval(Policy_type), env, verbose=0, tensorboard_log=tf_log_file)
                #model.load(f'./model/{RL_algorithm}_grid_stage3_gg.pkl')
                print("start to learn: ", RL_algorithm)
                model.learn(total_timesteps=int(5e5), callback=callback)
                model.save(model_file)
                model.save(f'./model/{RL_algorithm}_grid_stage3_gg.pkl')
                #plot_results(log_dir)

            except:
                print("failed")
