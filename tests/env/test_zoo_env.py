from pettingzoo.test import api_test
from yaniv_rl.envs import zoo_yaniv

env = zoo_yaniv.env()
api_test(env, num_cycles=10, verbose_progress=True)
