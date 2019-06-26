import gym
import envs
import numpy as np
import cv2

env = gym.make('CarRacing-v1')

for i_episode in range(20):
    observation = env.reset()
    #image=cv2.imshow('test',observation)
    for t in range(100):
        env.render()
        #print(env.action_space.sample(),'\n==============\n')

        sample = env.action_space.sample()
        sample_speed_weight = np.random.uniform(0, 25.0)
        sample_steer_weight = np.random.uniform(-0.5,0.5)
        action = [sample_steer_weight, sample_speed_weight, 0]
        print(action)
        #action = [0,5.5,0.0]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
