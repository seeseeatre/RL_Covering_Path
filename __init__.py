from gym.envs.registration import register

register(
    id='CarRacing-v1',
    entry_point='envs.car_racing1:CarRacing1',
    #max_episode_steps=10000,
    #reward_threshold=9999,
)

