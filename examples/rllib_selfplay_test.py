import ray
from ray import tune
from ray.tune import register_env
from ray.tune.trial import ExportFormat
from ray.tune.utils.log import Verbosity
from ray.tune.integration.wandb import WandbLoggerCallback

from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo.ppo import PPOTrainer

import argparse
import numpy as np

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.utils.rllib import YanivActionMaskModel, YanivCallbacks, make_eval_func, YanivTrainer

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
    "negative_score_cutoff": 20,
    "single_step": False,
    "step_reward": 0,
    "use_unkown_cards_in_state": False,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--eval-num", type=int, default=200)
    parser.add_argument("--eval-int", type=int, default=5)
    parser.add_argument("--random-players", type=int, default=0)
    parser.add_argument("--restore", type=str, default="")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--name", type=str, default="")

    args = parser.parse_args()

    print("cuda: ")
    cuda_avail()
    ray.init(
        local_mode=False,
    )

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space

    # def explore(config):
    #     # ensure we collect enough timesteps to do sgd
    #     if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
    #         config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    #     # ensure we run at least one sgd iter
    #     if config["num_sgd_iter"] < 1:
    #         config["num_sgd_iter"] = 1
    #     return config

    # from ray.tune.schedulers import PopulationBasedTraining
    # pbt = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="evaluation/eval_rules_win_rate",
    #     mode="max",
    #     perturbation_interval=10,
    #     resample_probability=0.25,
    #     # Specifies the mutations of these hyperparams
    #     hyperparam_mutations={
    #         "lambda": lambda: random.uniform(0.9, 1.0),
    #         "clip_param": lambda: random.uniform(0.01, 0.5),
    #         "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    #         "num_sgd_iter": lambda: random.randint(1, 30),
    #         "sgd_minibatch_size": lambda: random.randint(128, 16384),
    #         "train_batch_size": lambda: random.randint(2000, 160000),
    #     },
    #     custom_explore_fn=explore)

    config = {
        "env": "yaniv",
        "env_config": env_config,
        "model": {
            "custom_model": "yaniv_mask",
            "fcnet_hiddens": [512, 512],
            # "vf_share_layers": True,
        },
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
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_1"],
        },
        "callbacks": YanivCallbacks,
        "log_level": "INFO",
        "evaluation_num_workers": 0,
        "evaluation_config": {"explore": False},
        "evaluation_interval": args.eval_int,
        "custom_eval_function": make_eval_func(env_config, args.eval_num),
        # hyper params
        "batch_mode": "complete_episodes",
        "train_batch_size": 32768,
        "num_sgd_iter": 20,
        "sgd_minibatch_size": 2048,
        # for pbt
        # # These params are tuned from a fixed starting value.
        # "lambda": 0.95,
        # "clip_param": 0.2,
        # "lr": 1e-4,
        # # These params start off randomly drawn from a set.
        # "num_sgd_iter": tune.choice([10, 20, 30]),
        # "sgd_minibatch_size": tune.choice([128, 512, 2048]),
        # "train_batch_size": tune.choice([10000, 20000, 40000])
    }

    # from ray.tune.utils import validate_save_restore
    # print("check save restore")
    # print(validate_save_restore(YanivTrainer, config=config, num_gpus=0.5))
    # print("checked save restores")

    resources = PPOTrainer.default_resource_request(config)

    results = tune.run(
        YanivTrainer,
        resources_per_trial=resources,
        # scheduler=pbt,
        # num_samples=2,
        name=args.name,
        config=config,
        stop={"training_iteration": 1000},
        checkpoint_freq=5,
        checkpoint_at_end=True,
        verbose=Verbosity.V3_TRIAL_DETAILS,
        callbacks=[
            WandbLoggerCallback(
                project="rllib_yaniv",
                # api_key_file="/home/jippo/.netrc",
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
