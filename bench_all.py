import os
import argparse
import gym
import envs
import numpy as np
import matplotlib.pyplot as plt
import csv

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

if __name__ == "__main__":
    #RL_types = ["DQN", "TRPO", "A2C", "PPO2"]
    RL_types = ["DQN"]
    
    for RL_algorithm in RL_types:
        if RL_algorithm == "DQN":
            Policy_type = "DQMlpPolicy"
        else:
            Policy_type = "MlpPolicy"
            #Policy_type = "CnnPolicy"
        nattempts = 1
        os.makedirs("./ben_log/grid/", exist_ok=True)
        csv_log_file = f"./ben_log/grid/{RL_algorithm}_{Policy_type}_3_gg2.csv"
        csvFile = open(csv_log_file, "w")
        writer = csv.writer(csvFile)
        writer.writerow(["index","Reward","Episode Length","Action"])

        for attempt in range(1, nattempts+1):
            print(f'{RL_algorithm}_{Policy_type}_GRD{attempt}')

            
            env = gym.make('CarRacing-v2')
            env = DummyVecEnv([lambda: env])

            model = eval(RL_algorithm)(eval(Policy_type), env, verbose=0)
            #model._load_from_file(f'./model/{RL_algorithm}_grid_stage3_gg2.pkl')
            model._load_from_file(f'./model/DQN_grid_stage3_gg_best.pkl')
            
            print("start to bench: ", RL_algorithm)

            obs = env.reset()
            reward_sum = 0
            action_history = np.zeros(6)
            action_sum = np.zeros(6)
            ep_length = 0
            ep_length_record = []
            while len(ep_length_record) < 50:
                action = model.predict(obs)
                action_history[action]+=1
                action_sum[action]+=1
                obs, rewards, dones, info = env.step(action)
                reward_sum += rewards
                ep_length += 1
                #env.render()

                if dones:
                    writer.writerow([len(ep_length_record), reward_sum[0], ep_length, action_history/ep_length])

                    print("Ep No.: ", len(ep_length_record))

                    print("reward: ", reward_sum)
                    reward_sum = 0

                    print("actions: ", action_history)
                    action_history = np.zeros(6)

                    print("Episode length: ", ep_length)
                    ep_length_record.append(ep_length)
                    ep_length = 0

                    obs = env.reset()
            # except:
            #     print("failed")
        csvFile.close()
