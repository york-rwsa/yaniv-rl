from yaniv_rl.models.yaniv_rule_models import YanivIntermediateRuleAgent, YanivNoviceRuleAgent
from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv

def extract_rlcard_state(game, player):
    state = game.get_state(player)
    extracted_state = {}
    extracted_state["raw_obs"] = state
    extracted_state["raw_legal_actions"] = state["legal_actions"]

    return extracted_state

def intermediate_rule_step(env: YanivEnv):
    if env.single_step:
        raise NotImplementedError
    
    state = extract_rlcard_state(env.game, env.current_player)
    action = YanivIntermediateRuleAgent._step_multi(state)
    env.game.step(action)

def novice_rule_step(env: YanivEnv):
    state = extract_rlcard_state(env.game, env.current_player)

    if env.single_step:
        action = YanivNoviceRuleAgent._step_single(state)
    else:
        action = YanivNoviceRuleAgent._step_multi(state)
        
    env.game.step(action)
    