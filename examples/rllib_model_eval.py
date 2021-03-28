from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.policy import policy
from ray.rllib.utils.typing import AgentID
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
    parser.add_argument("--eval-num", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="")
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
        },
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 0,
        "batch_mode": "complete_episodes",
        "log_level": "INFO",
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
    }

    best_checkpoint = "/home/jippo/ray_results/YanivTrainer_2021-03-28_13-22-27/YanivTrainer_yaniv_4258f_00000_0_2021-03-28_13-22-27/checkpoint_340/checkpoint_340/checkpoint-340"

    config["explore"] = False
    agent = ppo.PPOTrainer(config=config, env="yaniv")
    agent.restore(best_checkpoint)

    rule_agent = YanivNoviceRuleAgent(single_step=True)

    env = YanivEnv(env_config)

    wins = 0
    draws = 0
    for _ in range(args.eval_num):
        episode_reward = 0
        done = {"__all__": False}
        obs = env.reset()

        agent_id = "player_0"
        rules_id = "player_1"

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
                obs, reward, done, info = env.step({rules_id: action}, raw_action=True)

            # episode_reward += reward[agent_id]
            steps += 1

        print(episode_reward, steps, reward)

        if reward[agent_id] == 0:
            draws += 1
        elif reward[agent_id] == 1:
            wins += 1

    print("Wins: {}, Draws: {}, Episodes: {}".format(wins, draws, args.eval_num))