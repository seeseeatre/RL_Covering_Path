from gym.envs.registration import registry, register, make, spec

# Box2d
# ----------------------------------------

register(
    id='CarRacing-v1',
    entry_point='envs.car_racing1:CarRacing1',
    # max_episode_steps=10000,
    # reward_threshold=9999,
),
register(
    id='CarRacing-v2',
    entry_point='envs.car_racing2:CarRacing2',
    # max_episode_steps=10000,
    # reward_threshold=9999,
)
