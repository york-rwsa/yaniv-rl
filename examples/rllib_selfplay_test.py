from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from ray.tune.utils.log import Verbosity
from yaniv_rl.utils.utils import ACTION_SPACE
import ray
import torch
import numpy as np
from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray import tune
from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents import ppo
import argparse

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

from ray.tune.integration.wandb import WandbLoggerCallback

torch, nn = try_import_torch()

# https://github.com/ray-project/ray/blob/739f6539836610e3fbaadd3cf9ad7fb9ae1d79f9/rllib/examples/models/parametric_actions_model.py
class YanivActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        true_obs_space = Box(low=0, high=1, shape=(266,), dtype=int)
        self.action_model = TorchFC(
            true_obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]

        action_logits, _ = self.action_model({"obs": input_dict["obs"]["state"]})

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return torch.reshape(self.action_model.value_function(), [-1])


class YanivCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        pass

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data
        """
        # Get env refernce from rllib wraper
        # env = base_env.get_unwrapped()[0]

        final_rewards = {k: r[-1] for k, r in episode._agent_reward_history.items()}

        episode.custom_metrics["final_reward"] = final_rewards["player_0"]
        episode.custom_metrics["win"] = 1 if final_rewards["player_0"] == 1 else 0
        episode.custom_metrics["draw"] = 1 if final_rewards["player_0"] == 0 else 0

    def on_sample_end(self, worker, samples, **kwargs):
        pass

    def on_train_result(self, trainer, result, **kwargs):
        pass

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        pass


def policy_mapping_fn(agent_id):
    if agent_id.endswith("0"):
        return "policy_1"  # Choose 1 policy for agent_0
    else:
        # trains against past versions of self
        return np.random.choice(
            ["policy_1", "policy_2", "policy_3", "policy_4"],
            p=[0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3],
        )


def copy_weights(to_policy, from_policy, trainer):
    """copy weights from from_policy to to_policy without changing from_policy"""
    temp_weights = {}  # temp storage with to_policy keys & from_policy values
    for (k, v), (k2, v2) in zip(
        trainer.get_policy(to_policy).get_weights().items(),
        trainer.get_policy(from_policy).get_weights().items(),
    ):
        temp_weights[k] = v2

    # set weights
    trainer.set_weights(
        {
            to_policy: temp_weights,  # weights or values from from_policy with to_policy keys
        }
    )

    # To check
    for (k, v), (k2, v2) in zip(
        trainer.get_policy(to_policy).get_weights().items(),
        trainer.get_policy(from_policy).get_weights().items(),
    ):
        assert (v == v2).all()

    print("{} == {}".format(to_policy, from_policy))


def shift_policies(trainer, new, p2, p3, p4):
    copy_weights(p4, p3, trainer)
    copy_weights(p3, p2, trainer)
    copy_weights(p2, new, trainer)


def make_eval_func(env_config, eval_num):
    def yaniv_eval(trainer, eval_workers):
        print("\n\n\n************** EVALUATION **************")
        
        agent = trainer
        rule_agent = YanivNoviceRuleAgent(single_step=True)
        agent_id = "player_0"
        rules_id = "player_1"

        env = YanivEnv(env_config)

        wins = 0
        draws = 0
        total_steps = 0
        for _ in range(eval_num):
            episode_reward = 0
            done = {"__all__": False}
            obs = env.reset()

            steps = 0
            while not done["__all__"]:
                if env.current_player == 0:
                    action = agent.compute_action(obs[agent_id], policy_id="policy_1")
                    obs, reward, done, info = env.step({agent_id: action})
                else:
                    state = env.game.get_state(1)
                    extracted_state = {}
                    extracted_state["raw_obs"] = state
                    extracted_state["raw_legal_actions"] = [
                        a for a in state["legal_actions"]
                    ]

                    action = rule_agent.step(extracted_state)
                    obs, reward, done, info = env.step(
                        {rules_id: action}, raw_action=True
                    )

                steps += 1

            # print(episode_reward, steps, reward)

            # metrics
            if reward[agent_id] == 0:
                draws += 1
            elif reward[agent_id] == 1:
                wins += 1
            total_steps += steps

        eval_vs = "eval_rules_"
        metrics = {
            eval_vs + "draw_rate": draws / eval_num,
            eval_vs + "avg_roundlen": total_steps / eval_num,
            eval_vs + "win_rate": wins / eval_num,
        }

        print(pretty_print(metrics), "\n\n\n")

        return metrics

    return yaniv_eval


class YanivTrainer(tune.Trainable):
    def setup(self, config):
        self.trainer = PPOTrainer(env="yaniv", config=config)
        self.config = config

    def step(self):
        result = self.trainer.train()

        if result["custom_metrics"]["win_mean"] > 0.55:
            shift_policies(self.trainer, "policy_1", "policy_2", "policy_3", "policy_4")
            print("weights shifted")

        return result

    def save_checkpoint(self, dir):
        self.trainer.save(dir)
        return dir

    def load_checkpoint(self, checkpoint):
        self.trainer.load_checkpoint(checkpoint)


def cuda_avail():
    import torch

    print(torch.cuda.is_available())


env_config = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 130,
    "early_end_reward": 0,
    "use_scaled_negative_reward": False,
    "max_negative_reward": -1,
    "negative_score_cutoff": 50,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--eval-num", type=int, default=1)
    parser.add_argument("--random-players", type=int, default=0)
    parser.add_argument("--restore", type=str, default="")
    args = parser.parse_args()

    print("cuda: ")
    cuda_avail()
    ray.init(local_mode=True)

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space
    config = {
        "env": "yaniv",
        "env_config": env_config,
        "model": {
            "custom_model": "yaniv_mask",
            "fcnet_hiddens": [512, 512],
            # "vf_share_layers": True,
        },
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": args.num_workers,
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
        "batch_mode": "complete_episodes",
        "log_level": "INFO",
        "evaluation_num_workers": 0,
        "evaluation_config": {"explore": False},
        "evaluation_interval": 5,
        "custom_eval_function": make_eval_func(env_config, 500)
        # "lr": tune.grid_search([0.0005, 1e-5]),
    }

    resources = PPOTrainer.default_resource_request(config).to_json()
    tune.run(
        YanivTrainer,
        resources_per_trial=resources,
        config=config,
        stop={"training_iteration": 1000},
        checkpoint_freq=20,
        checkpoint_at_end=True,
        verbose=Verbosity.V3_TRIAL_DETAILS,
        callbacks=[
            WandbLoggerCallback(
                project="rllib_yaniv",
                # api_key_file="/home/jippo/.netrc",
                log_config=True,
            )
        ],
        restore=args.restore
    )
