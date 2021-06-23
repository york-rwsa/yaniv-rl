from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel


torch, nn = try_import_torch()


class YanivActionMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        true_obs_space = Box(
            low=0, high=1, shape=obs_space.original_space["state"].shape, dtype=int
        )
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
        
class DQNYanivActionMaskModel(DQNTorchModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        true_obs_space = Box(
            low=0, high=1, shape=obs_space.original_space["state"].shape, dtype=int
        )
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
