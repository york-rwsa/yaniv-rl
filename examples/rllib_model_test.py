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

torch, nn = try_import_torch()


# for available actions check out:
# https://github.com/ray-project/ray/blob/739f6539836610e3fbaadd3cf9ad7fb9ae1d79f9/rllib/examples/models/parametric_actions_model.py
class YanivModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        true_obs_space = Box(low=0, high=1, shape=(266,), dtype=int)
        self.torch_sub_model = TorchFC(true_obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        
        action_logits, _ = self.torch_sub_model({
            "obs": input_dict["obs"]["state"] # NOTE: maybe need to add .float()?
        })
        
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        # import pdb; pdb.set_trace()
        return action_logits + inf_mask, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

env_config = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 100,
    "early_end_reward": -1,
    "use_scaled_negative_reward": True,
    "max_negative_reward": -1,
    "negative_score_cutoff": 50,
}

eval_config = {
    'env_config': env_config
}


num_episodes_per_scenario = 20
eval_metrics = []
def eval_func(trainer, eval_workers):
    """Evaluates the performance of the domray model by playing it against
    Provincial using preset buy menus.
    Args:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict
    """
    global eval_metrics

    for i in range(num_episodes_per_scenario):
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=600)

    metrics = summarize_episodes(episodes)
    eval_metrics.append(metrics)

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    ray.init(local_mode=True)

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("yaniv_mask", YanivModel)
    
    config = {
        "env_config": env_config,
        "model": {
            "custom_model": "yaniv_mask",
        },
        "framework": "torch",
    }

    stop = {
        "training_iteration": args.num_iters
    }

    trainer = ppo.PPOTrainer(env='yaniv', config=config)
    res = trainer.train()
    print(res)
    import pdb; pdb.set_trace()

    