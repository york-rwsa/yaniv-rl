import ray
from ray import tune
from ray.rllib.agents.registry import get_trainer_class
from ray.tune import register_env
from ray.tune.trial import ExportFormat
from ray.tune.utils.log import Verbosity
from ray.tune.integration.wandb import WandbLoggerCallback

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer

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


def cuda_avail():
    import torch

    print(torch.cuda.is_available())


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
    "state_n_players": 2
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--eval-num", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--random-players", type=int, default=0)
    parser.add_argument("--restore", type=str, default="")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--address", type=str, default=None)
    
    args = parser.parse_args()

    print("cuda: ")
    cuda_avail()

    ray.init(
        local_mode=False,
        address=args.address
    )

    register_env("yaniv", lambda config: YanivEnv(config))

    from yaniv_rl.utils.rllib.model import DQNYanivActionMaskModel
    
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space

    config = {
        "env": "yaniv",
        "env_config": env_config,
        "framework": "torch",
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        # "num_envs_per_worker": 1,
        # "num_cpus_per_worker": 1,
        # "num_cpus_for_driver": 1,
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
        "callbacks": YanivCallbacks,
        "log_level": "INFO",
        "evaluation_num_workers": 0,
        "evaluation_config": {"explore": False},
        "evaluation_interval": args.eval_every,
        "custom_eval_function": make_eval_func(env_config, args.eval_num),
        
        # hyper params
        "model": {
            "custom_model": "yaniv_mask",
            "fcnet_hiddens": [],
        },
        "batch_mode": "complete_episodes",

        # A3c
        # "algorithm": "A3C",
        # "rollout_fragment_length": 50,
        # "train_batch_size": 500,
        # "min_iter_time_s": 10,
        
        # ppo
        # "algorithm": "PPO",
        # "sgd_minibatch_size": 2048,
        # "train_batch_size": 65536,
        # "rollout_fragment_length": 100

        # dqn
        "algorithm": "APEX",
        "train_batch_size": 4096,
        "dueling": False,
        "hiddens": [],
        ""
        # "double_q": False,
    }

    trainer_class = get_trainer_class(config["algorithm"])
    # config.pop("algorithm")
    # trainer = trainer_class(env="yaniv", config=config)
    # trainer.train()
    
    resources = trainer_class.default_resource_request(config)

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

