from ray.tune.logger import pretty_print
from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent, YanivIntermediateRuleAgent

import pickle5 as pickle
from yaniv_rl.utils.rllib.tournament import YanivTournament
import argparse
import ray
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.a3c.a3c import A3CTrainer
import numpy as np

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.utils.rllib import (
    YanivActionMaskModel,
    YanivCallbacks,
    make_eval_func,
)

def step_func(env: YanivEnv):
    state = env.game.get_state(env.current_player)
    extracted_state = {}
    extracted_state["raw_obs"] = state
    extracted_state["raw_legal_actions"] = state["legal_actions"]

    action = YanivIntermediateRuleAgent._step_multi(extracted_state)
    env.game.step(action)
    # print(action)

def policy_mapping_fn(agent_id):
    # if agent_id == "player_1":
    #     return None

    return "policy_1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-num", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--address", type=str, default=None)
    args = parser.parse_args()

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    env_config = {
        "end_after_n_deck_replacements": 0,
        "end_after_n_steps": 130,
        "early_end_reward": 0,
        "use_scaled_negative_reward": True,
        "use_scaled_positive_reward": True,
        "max_negative_reward": -1,
        "negative_score_cutoff": 30,
        "single_step": False,
        "step_reward": 0,
        "use_unkown_cards_in_state": False,
        "use_dead_cards_in_state": True,
        "observation_scheme": 1,
        "n_players": 2,
        "state_n_players": 2,
        "player_step_fn": {
            "player_0": step_func,
            # "player_1": step_func
        }
    }

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space

    config = {
        "callbacks": YanivCallbacks,
        "num_gpus": 1,
        "env": "yaniv",
        "env_config": env_config,
        "framework": "torch",
        "multiagent": {
            "policies": {
                "policy_1": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_1"],

        },
        "model": {
            "custom_model": "yaniv_mask",
            "fcnet_hiddens": [512, 512],
        },

        "num_workers": 1,
        "evaluation_num_workers": args.num_workers,
        "evaluation_num_episodes": args.eval_num,
        "evaluation_interval": 1,
    }

    ray.init(
        include_dashboard=False, 
        local_mode=True
    )

    model = "/home/jippo/ray_results/YanivTrainer_2021-05-02_16-32-35/YanivTrainer_yaniv_9e76a_00000_0_2021-05-02_16-32-35/models/model-4560.pkl"
    with open(model, 'rb') as f:
        policy = pickle.load(f)

    trainer = A3CTrainer(env="yaniv", config=config)
    trainer.get_policy("policy_1").set_state(policy)


    metrics = trainer._evaluate()
    metrics["evaluation"].pop("hist_stats")
    print(pretty_print(metrics))