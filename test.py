import gym
import numpy as np
env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


best_length =0
episode_lengths = []
best_weights = np.zeros(4)

for i in range(100):
    new_weights = np.random.uniform(-1.0, 1.0, 4)

    length = []

    for i_episode in range(100):
        observation = env.reset()
        done = False
        cnt =0
        while not done:
            #env.render()
            cnt +=1
            #print(observation)
            #action = env.action_space.sample()
            action = 1 if np.dot(observation, new_weights)>0 else 0
            observation, reward, done, info = env.step(action)
            if done:
                #print("Episode finished after {} timesteps".format(cnt+1))
                break
        length.append(cnt)
    avg_length = float(sum(length)/len(length))

    if avg_length > best_length:
        best_length = avg_length
        best_weights = new_weights
    episode_lengths.append(avg_length)
    if i % 10 ==0:
        print('best length is ', best_length)

done = False
cnt=0
observation = env.reset()
print("start rendering with best weight")
while not done:
    env.render()
    cnt+=1
    action = 1 if np.dot(observation, best_weights)>0 else 0
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(cnt))
        break 



env.close()
