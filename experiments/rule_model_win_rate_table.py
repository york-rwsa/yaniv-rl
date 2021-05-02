from itertools import combinations, combinations_with_replacement
import rlcard
from rlcard.utils import set_global_seed
from rlcard.agents import RandomAgent
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent, YanivIntermediateRuleAgent
from yaniv_rl import utils
from rlcard.envs.registration import register, make

register(
    env_id='yaniv',
    entry_point='yaniv_rl.envs.yaniv:YanivEnv',
)


agents = [RandomAgent(488), YanivNoviceRuleAgent(), YanivIntermediateRuleAgent()]

# Make environment
env = make('yaniv', config={'seed': 0})



eval_num = 10000

table = [[0 for i in range(3)] for i in range(3)]
for i in range(3):
    # player v player
    env.set_agents([agents[i], agents[i]])
    res = utils.tournament(env, eval_num)
    winrate = res['wins'][0] / eval_num
    table[i][i] = winrate

for agent_1, agent_2 in combinations(agents, 2):
    a1i = agents.index(agent_1)
    a2i = agents.index(agent_2)
    env.set_agents([agent_1, agent_2])
    res = utils.tournament(env, eval_num)
    table[a1i][a2i] = res['wins'][0] / eval_num
    table[a2i][a1i] = res['wins'][1] / eval_num
    
    
print(table)
