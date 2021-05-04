from itertools import combinations, combinations_with_replacement
import json
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


intagent = YanivIntermediateRuleAgent()
eval_num = 10000

with open("./hand_classes.json") as f:
    handclasses = json.load(f)

results = {}

for handclass, hands in handclasses.items():
    env = make('yaniv', config={'seed': 0, 'starting_hands': {0: hands}})
    env.set_agents([intagent, intagent])
    res = utils.tournament(env, eval_num)
    
    print("**** HAND CLASS: {} ****".format(handclass))
    print(res)
    print()
    results[handclass] = res

print(results)

with open("staring_hand_class_rules_output.json", "w") as f:
    json.dump(results, f, indent=4)