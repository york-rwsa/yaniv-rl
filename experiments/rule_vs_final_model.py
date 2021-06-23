import copy
import json
import logging
import random
from tqdm import tqdm
import math
import os
import pickle5 as pickle
from ray.rllib.env.env_context import EnvContext
from tqdm.std import trange
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
    make_eval_func,
    intermediate_rule_step,
    novice_rule_step,
)

logger = logging.getLogger(__name__)


def policy_mapping_fn(agent_id):
    return "policy_1"


def eval_fn(trainer, env_config, hands) -> dict:
    """Evaluates current policy under `evaluation_config` settings.

    Note that this default implementation does not do anything beyond
    merging evaluation_config with the normal trainer config.
    """
    # Call the `_before_evaluate` hook.
    trainer._before_evaluate()
    # Sync weights to the evaluation WorkerSet.
    trainer._sync_weights_to_workers(worker_set=trainer.evaluation_workers)
    trainer._sync_filters_if_needed(trainer.evaluation_workers)
    
    if trainer.config["evaluation_num_workers"] == 0:
        for _ in range(trainer.config["evaluation_num_episodes"]):
            trainer.evaluation_workers.local_worker().sample()
    else:
        num_rounds = int(
            math.ceil(trainer.config["evaluation_num_episodes"] /
                        trainer.config["evaluation_num_workers"]))
        num_workers = len(trainer.evaluation_workers.remote_workers())
        num_episodes = num_rounds * num_workers
        
        for i in trange(0, num_episodes, num_workers, unit_scale=num_workers, leave=False):
            update_config()
            logger.info("Running round {} of parallel evaluation "
                        "({}/{} episodes)".format(
                            i, (i + 1) * num_workers, num_episodes))

            ray.get([
                w.sample.remote()
                for w in trainer.evaluation_workers.remote_workers()
            ])

        metrics = collect_metrics(trainer.evaluation_workers.local_worker(),
                                    trainer.evaluation_workers.remote_workers())
    return {"evaluation": metrics}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-num", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--cpus-for-driver", type=float, default=0.5)
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/jippo/ray_results/YanivTrainer_2021-05-02_16-44-14/YanivTrainer_yaniv_3ee8a_00000_0_2021-05-02_16-44-14/models/model-010000.pkl",
    )
    parser.add_argument("--handclasses-path", type=str, default="../hand_classes.json")
    parser.add_argument("--opponent", type=str, default="intermediate")
    args = parser.parse_args()

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    if args.opponent == "intermediate":
        stepfn = intermediate_rule_step
    elif args.opponent == "novice":
        stepfn = novice_rule_step
    else:
        raise ValueError("opponent not defined: {}".format(args.opponent))

    with open(args.handclasses_path) as f:
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
        "player_step_fn": {"player_1": stepfn},
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
        # "log_level": "DEBUG"
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": 1,
    }

    ray.init(include_dashboard=False, local_mode=True, logging_level=logging.INFO)

    trainer = A3CTrainer(env="yaniv", config=config)
    with open(args.model_path, "rb") as f:
        policy = pickle.load(f)
    trainer.get_policy("policy_1").set_state(policy)

    results = []

    def make_update_env_fn(env_conf):
        def update_env_conf(env):
            env.config.update(env_conf)
            env.game.configure(env.config)

        def update_env_fn(worker):
            worker.foreach_env(update_env_conf)

        return update_env_fn

    handclasses['NONE'] = []

    try:
        for handclass, hands in tqdm(handclasses.items()):
            for i in range(2):
                # config["env_config"]["starting_hands"] = {i: hands}

                # trainer.evaluation_workers.foreach_worker(
                #     make_update_env_fn(config["env_config"])
                # )

                # metrics = trainer._evaluate()
                handconf = {i: hands} if len(hands) > 0 else {}
                metrics = eval_fn(trainer, env_config, handconf)

                metrics["evaluation"].pop("hist_stats")

                stats = {
                    k: v
                    for k, v in metrics["evaluation"]["custom_metrics"].items()
                    if k.endswith("mean")
                }

                stats["handclass"] = handclass
                stats["player_with_class"] = i
                tqdm.write(
                    "handclass: {: <6}: player: {}, win_mean: {:.2f}, episodes: {}, len: {:.1f}".format(
                        handclass,
                        i,
                        stats["player_{}_win_mean".format(i)],
                        metrics["evaluation"]["episodes_this_iter"],
                        metrics['evaluation']['episode_len_mean']
                    )
                )
                results.append(stats)

        with open(
            "handclassoutput_vs_{}_{}.json".format(args.opponent, args.eval_num), "w"
        ) as f:
            json.dump(results, f, indent=4)
    finally:
        trainer.stop()