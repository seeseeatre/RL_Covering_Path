import os
import gym
import envs
import numpy as np
import matplotlib.pyplot as plt
import random
import csv

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, MlpLstmPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

csvFile = open("random_agent_log.csv", "w")
writer = csv.writer(csvFile)
writer.writerow(["index","Reward","Episode Length","Action"])

# Create and wrap the environment
env = gym.make('CarRacing-v1')
env = DummyVecEnv([lambda: env])

obs = env.reset()
reward_sum = 0
action_history = np.zeros(6)
action_sum = np.zeros(6)
ep_length = 0
ep_length_record = []
while len(ep_length_record) < 50:
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

if ep_length >0:
  ep_length_record.append(ep_length)

print("average actions per episode: ", action_sum/len(ep_length_record))
ep_length_avg = sum(ep_length_record) / len(ep_length_record)
print("Average Episode Length: ", ep_length_avg)

csvFile.close()
