from rlcard.envs.registration import register, make

register(
    env_id='yaniv',
    entry_point='yaniv_rl.envs.yaniv:YanivEnv',
)
