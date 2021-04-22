from yaniv_rl.utils.rllib.tournament import YanivTournament
import ray

from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.a3c.a3c import A3CTrainer
import argparse
import numpy as np

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.utils.rllib import (
    YanivActionMaskModel,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-num", type=int, default=200)
    parser.add_argument("--ppo-checkpoint", type=str, default="/home/jippo/ray_results/YanivTrainer_2021-04-03_21-40-03/YanivTrainer_yaniv_c49f4_00000_0_2021-04-03_21-40-03/checkpoint_001580/checkpoint-225")
    parser.add_argument("--a3c-checkpoint", type=str, default="/home/jippo/ray_results/YanivTrainer_2021-04-11_23-01-13/YanivTrainer_yaniv_6e345_00000_0_2021-04-11_23-01-13/checkpoint_021605/checkpoint-13385")
    parser.add_argument("--obs-scheme", type=int, default=0)
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
        "observation_scheme": args.obs_scheme,
        "n_players": 2,
        "state_n_players": 2,
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

    ray.init(include_dashboard=False, local_mode=True)

    ppo = PPOTrainer(env="yaniv", config=config)
    ppo.restore(args.ppo_checkpoint)

    a3c = A3CTrainer(env="yaniv", config=config)
    a3c.restore(args.a3c_checkpoint)

    tourney = YanivTournament(env_config, trainers=[a3c], opponent="intermediate")
    # tourney.run_episode(True)
    # tourney.print_stats()
    tourney.run(args.eval_num)
    print("\n\nRESULTS:\n")
    tourney.print_stats()