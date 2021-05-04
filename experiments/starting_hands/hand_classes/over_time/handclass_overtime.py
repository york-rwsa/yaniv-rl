import copy
import json
import logging
from tqdm import tqdm
import math
import os
import pickle5 as pickle
from ray.rllib.env.env_context import EnvContext
from yaniv_rl import utils
from ray.rllib.agents.callbacks import DefaultCallbacks

from ray.rllib.evaluation.metrics import collect_metrics, summarize_episodes
from ray.tune.logger import pretty_print

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
    YanivCallbacks,
)


logger = logging.getLogger(__name__)


def policy_mapping_fn(agent_id):
    return "policy_1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-num", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--cpus-for-driver", type=float, default=0.5)
    args = parser.parse_args()

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    with open("../hand_classes.json") as f:
        handclasses = json.load(f)

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
        "starting_player": "random",
        "starting_hands": {},
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
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": args.cpus_per_worker,
        "num_cpus_for_driver": args.cpus_for_driver,
        "num_workers": 1,
        "evaluation_num_workers": args.num_workers,
        "evaluation_num_episodes": args.eval_num,
        "evaluation_interval": 1,
    }

    ray.init(include_dashboard=False, local_mode=False)

    trainer = A3CTrainer(env="yaniv", config=config)
    # models_path = "/scratch/student/models"
    models_path = "/home/jippo/ray_results/YanivTrainer_2021-05-02_16-44-14/YanivTrainer_yaniv_3ee8a_00000_0_2021-05-02_16-44-14/models"
    models = os.listdir(models_path)

    results = {}

    def make_update_env_fn(env_conf):
        def update_env_conf(env):
            env.config.update(env_conf)
            env.game.configure(env.config)
            
        def update_env_fn(worker):
            worker.foreach_env(update_env_conf)

        return update_env_fn

    try:
        for model in tqdm(sorted(models)):
            if not model.startswith("model"):
                print("idk", model)
                continue

            model_num = int(model[6:-4])
            model_results = []
            if model_num % 50 != 0:
                continue
            print("\n**** MODEL NUM: {} ****".format(model_num))
            path = os.path.join(models_path, model)
            with open(path, 'rb') as f:
                policy = pickle.load(f)
            
            trainer.get_policy("policy_1").set_state(policy)

            for handclass, hands in handclasses.items():
                if len(hands) < 1:
                    continue

                config["env_config"]["starting_hands"] = {0: hands}

                # print("**** updating env config: starting_hand_score: {} ****".format(handclass))
                trainer.evaluation_workers.foreach_worker(
                    make_update_env_fn(config["env_config"])
                )
                
                metrics = trainer._evaluate()

                metrics["evaluation"].pop("hist_stats")

                stats = {
                    k: v
                    for k, v in metrics["evaluation"]["custom_metrics"].items()
                    if k.endswith("mean")
                }

                stats["handclass"] = handclass
                print("handclass: {0: <5}".format(handclass), "win_mean: ", stats["player_0_win_mean"])
                model_results.append(stats)
            
            results[str(model_num)] = model_results


        with open("handclassoutput.json", "w") as f:
            json.dump(results, f, indent=4)
    finally:
        trainer.stop()