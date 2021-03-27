import ray
import torch
import numpy as np
from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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


torch, nn = try_import_torch()


# for available actions check out:
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


env_config = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 100,
    "early_end_reward": 0,
    "use_scaled_negative_reward": False,
    "max_negative_reward": -1,
    "negative_score_cutoff": 50,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-num", type=int, default=1)
    parser.add_argument("--random-players", type=int, default=0)
    args = parser.parse_args()

    ray.init(local_mode=True)

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivActionMaskModel)

    config = {
        "env": "yaniv",
        "env_config": env_config,
        "model": {
            "custom_model": "yaniv_mask",
        },
        "framework": "torch",
        "num_gpus": 1
    }

    stop = {"training_iteration": args.num_iters}

    if args.train:
        config["num_workers"] = args.num_workers
        results = tune.run("PPO", config=config, stop=stop, checkpoint_at_end=True)
        best_checkpoint = results.get_best_checkpoint(
            trial=results.get_best_trial(metric="episode_reward_mean", mode="max"),
            metric="episode_reward_mean",
            mode="max",
        )
    else:
        best_checkpoint = "/home/jippo/ray_results/PPO/PPO_yaniv_ca7c0_00000_0_2021-03-27_11-37-20/checkpoint_10/checkpoint-10"

    agent = ppo.PPOTrainer(config=config, env="yaniv")
    agent.restore(best_checkpoint)

    rule_agent = YanivNoviceRuleAgent(single_step=True)

    env = YanivEnv(env_config)

    for _ in range(args.eval_num):
        episode_reward = 0
        done = {"__all__": False}
        obs = env.reset()

        agent_id = "player_0"
        rules_id = "player_1"

        steps = 0
        while not done["__all__"]:
            if env.current_player == 0:
                action = agent.compute_action(obs[agent_id])
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

            episode_reward += reward[agent_id]
            steps += 1
            
        print(episode_reward, steps, reward)

    ray.shutdown()