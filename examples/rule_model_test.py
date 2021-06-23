import rlcard
from rlcard.utils import set_global_seed
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent, YanivIntermediateRuleAgent
from yaniv_rl import utils
from rlcard.envs.registration import register, make

register(
    env_id='yaniv',
    entry_point='yaniv_rl.envs.yaniv:YanivEnv',
)


# Make environment
env = make('yaniv', config={'seed': 0})
episode_num = 2

# Set a global seed
set_global_seed(0)
env.set_agents([
    # YanivNoviceRuleAgent(),
    YanivIntermediateRuleAgent(),
    YanivIntermediateRuleAgent()
])

print(utils.tournament(env, 1000))

# for episode in range(episode_num):

#     # Generate data from the environment
#     trajectories, _ = env.run(is_training=False)

#     # Print out the trajectories
#     print('\nEpisode {}'.format(episode))
#     for ts in trajectories[0]:
#         print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.format(ts[0], ts[1], ts[2], ts[3], ts[4]))