import logging
import math
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


class YanivCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data
        """
        # Get env refernce from rllib wraper
        env = base_env.get_unwrapped()[0]

        metrics = {}
        for pid in env._get_players():
            metrics.update({
                "draw": 0,
                pid + "_win": 0,
                pid + "_assaf": 0
            })

        winner = env.game.round.winner
        if winner == -1:
            metrics["draw"] = 1
        else:
            winner_id = env._get_player_string(winner)
            metrics[winner_id + "_win"] = 1
            metrics[winner_id + "_winning_hands"] = utils.get_hand_score(
                env.game.players[winner].hand
            )

        assaf = env.game.round.assaf

        if assaf is not None:
            metrics[env._get_player_string(assaf) + "_assaf"] = 1

        s = env.game.round.scores
        if s is not None:
            for i in range(env.num_players):
                if s[i] > 0:
                    metrics[env._get_player_string(i) + "_losing_score"] = s[i]
        
        episode.custom_metrics.update(metrics)


def eval_func(trainer, workers):
    logger.info(
        "Evaluating current policy for {} episodes.".format(
            trainer.config["evaluation_num_episodes"]
        )
    )
    if trainer.config["evaluation_num_workers"] == 0:
        for _ in range(trainer.config["evaluation_num_episodes"]):
            trainer.evaluation_workers.local_worker().sample()
    else:
        num_rounds = int(
            math.ceil(
                trainer.config["evaluation_num_episodes"]
                / trainer.config["evaluation_num_workers"]
            )
        )
        num_workers = len(trainer.evaluation_workers.remote_workers())
        num_episodes = num_rounds * num_workers
        for i in range(num_rounds):
            logger.info(
                "Running round {} of parallel evaluation "
                "({}/{} episodes)".format(i, (i + 1) * num_workers, num_episodes)
            )
            ray.get(
                [w.sample.remote() for w in trainer.evaluation_workers.remote_workers()]
            )

    metrics = collect_metrics(
        trainer.evaluation_workers.local_worker(),
        trainer.evaluation_workers.remote_workers(),
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-num", type=int, default=200)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/jippo/Code/yaniv/yaniv-rl/examples/trained_models/A3C_36k_2player/checkpoint-15075",
    )
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
        "observation_scheme": 0,
        "n_players": 2,
        "state_n_players": 2,
        # "starting_player": ,
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

        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 0.5,
        "num_cpus_for_driver": 0.5,
        "num_workers": 1,
        "evaluation_num_workers": 6,
        "evaluation_num_episodes": args.eval_num,
        "evaluation_interval": 1
    }

    ray.init(include_dashboard=False)

    trainer = A3CTrainer(env="yaniv", config=config)
    trainer.restore(args.checkpoint)

    metrics = trainer._evaluate()
    metrics["evaluation"].pop("hist_stats")

    print(pretty_print(metrics))