from yaniv_rl.utils.rllib.tournament import YanivTournament
import ray
from ray import tune
from ray.tune import register_env
from ray.tune.trial import ExportFormat
from ray.tune.utils.log import Verbosity
from ray.tune.integration.wandb import WandbLoggerCallback

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.a3c.a3c import A3CTrainer

import argparse
import numpy as np

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.utils.rllib import (
    YanivActionMaskModel,
    YanivCallbacks,
    make_eval_func,
)


def policy_mapping_fn(agent_id):
    if agent_id.endswith("0"):
        return "policy_1"  # Choose 1 policy for agent_0
    else:
        # trains against past versions of self
        return np.random.choice(
            ["policy_1", "policy_2", "policy_3", "policy_4"],
            p=[0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3],
        )

if __name__ == "__main__":
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
        "observation_scheme": 0,
        "n_players": 2,
        "state_n_players": 3,
    }

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space

    config = {
        "num_gpus": 1,
        "env": "yaniv",
        "env_config": env_config,
        "framework": "torch",
        "multiagent": {
            "policies": {
                "policy_1": (None, obs_space, act_space, {}),
                "policy_2": (None, obs_space, act_space, {}),
                "policy_3": (None, obs_space, act_space, {}),
                "policy_4": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_1"],

        },
        "model": {
            "custom_model": "yaniv_mask",
            "fcnet_hiddens": [512, 512],
        },
    }

    ray.init(include_dashboard=False)

    checkpoint = "/home/jippo/ray_results/YanivTrainer_2021-04-14_10-29-11/YanivTrainer_yaniv_df2c8_00000_0_2021-04-14_10-29-12/checkpoint_032355/checkpoint-16350"
    

    trainer = A3CTrainer(env="yaniv", config=config)
    trainer.restore(checkpoint)

    t13k = A3CTrainer(env="yaniv", config=config)
    t13k.restore("/home/jippo/ray_results/YanivTrainer_2021-04-12_21-45-07/YanivTrainer_yaniv_f71d8_00000_0_2021-04-12_21-45-07/checkpoint_014760/checkpoint-14760")

    tourny = YanivTournament(env_config, trainers=[trainer, t13k])
    tourny.run(1000)
    print("\n\nRESULTS:\n")
    tourny.print_stats()