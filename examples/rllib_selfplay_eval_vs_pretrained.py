import os
import pickle5 as pickle
import ray
from ray import tune
from ray.tune import register_env
from ray.tune.trial import ExportFormat
from ray.tune.utils.log import Verbosity
from ray.tune.integration.wandb import WandbLoggerCallback

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.registry import get_trainer_class

import argparse
import numpy as np

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.utils.rllib import (
    YanivActionMaskModel,
    YanivCallbacks,
    make_eval_func,
    YanivTrainer,
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


def eval_mapping_fn(agent_id):
    if agent_id.endswith("0"):
        return "policy_1"
    else:
        return "eval_policy"


def cuda_avail():
    import torch

    print(torch.cuda.is_available())

def get_policy_weights_from_checkpoint(trainer_class, checkpoint):
    run_base_dir = os.path.dirname(checkpoint)
    config_path = os.path.join(run_base_dir, 'params.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    config.pop("algorithm")

    config['num_workers'] = 1
    config['evaluation_num_workers'] = 0
    eval_trainer = trainer_class(env="yaniv", config=config)
    eval_trainer.load_checkpoint(checkpoint)
    weights = eval_trainer.get_policy("policy_1").get_weights()
    eval_trainer.stop()
    
    return weights


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
    "state_n_players": 2,
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--eval-num", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--restore", type=str, default="")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--name", type=str, default="")

    args = parser.parse_args()

    print("cuda: ")
    cuda_avail()

    ray.init(
        local_mode=False,
        # address="auto"
    )

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space

    config = {
        "algorithm": "A3C",
        "env": "yaniv",
        "env_config": env_config,
        "framework": "torch",
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.5,
        "multiagent": {
            "policies": {
                "policy_1": (None, obs_space, act_space, {}),
                "policy_2": (None, obs_space, act_space, {}),
                "policy_3": (None, obs_space, act_space, {}),
                "policy_4": (None, obs_space, act_space, {}),
                "eval_policy": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_1"],
        },
        "callbacks": YanivCallbacks,
        
        "evaluation_num_workers": 0,
        "evaluation_config": {
            "explore": False,
            "multiagent": {"policy_mapping_fn": eval_mapping_fn},
        },
        "evaluation_interval": args.eval_every,
        "evaluation_num_episodes": args.eval_num,

        "update_self_play_param_win_rate": 0.55,

        # hyper params
        "model": {
            "custom_model": "yaniv_mask",
            "fcnet_hiddens": [512, 512],
        },
        
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": 100,
    }

    print("loading pretrained weights")
    trainer_class = get_trainer_class(config["algorithm"])
    resources = trainer_class.default_resource_request(config)
    weights = get_policy_weights_from_checkpoint(trainer_class, "/home/jippo/Code/yaniv/yaniv-rl/examples/trained_models/A3C_36k_2player/checkpoint-15075")
    print("weights loaded\n\n")


    config['evaluation_weights'] = weights
    
    results = tune.run(
        YanivTrainer,
        resources_per_trial=resources,
        name=args.name,
        config=config,
        stop={"training_iteration": 20000},
        checkpoint_freq=5,
        checkpoint_at_end=True,
        verbose=Verbosity.V3_TRIAL_DETAILS,
        callbacks=[
            WandbLoggerCallback(
                project="rllib_yaniv",
                log_config=True,
                id=args.wandb_id,
                resume="must" if args.wandb_id is not None else "allow",
            )
        ],
        export_formats=[ExportFormat.MODEL],
        restore=args.restore,
        keep_checkpoints_num=5,
        max_failures=8,
    )
