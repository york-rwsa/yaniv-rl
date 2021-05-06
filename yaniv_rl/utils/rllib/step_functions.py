from yaniv_rl.models.yaniv_rule_models import YanivIntermediateRuleAgent
from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv

def intermediate_rule_step(env: YanivEnv):
    state = env.game.get_state(env.current_player)
    extracted_state = {}
    extracted_state["raw_obs"] = state
    extracted_state["raw_legal_actions"] = state["legal_actions"]

    action = YanivIntermediateRuleAgent._step_multi(extracted_state)
    env.game.step(action)