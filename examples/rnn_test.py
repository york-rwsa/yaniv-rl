import ray
from ray import tune
from yaniv_rl.utils.rllib.tournament import YanivTournament

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MAX, FLOAT_MIN
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

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN

torch, nn = try_import_torch()

class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 fc_size=64,
                 lstm_state_size=256):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.num_outputs = num_outputs
        self.obs_size = int(np.product(obs_space.original_space["state"].shape))
        
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(TorchRNN)
    def forward(self, input_dict, state, seq_lens):
        flat_inputs = input_dict["obs"]["state"].float()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        action_logits, new_state = self.forward_rnn(inputs, state, seq_lens)
        action_logits = torch.reshape(action_logits, [-1, self.num_outputs])

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return action_logits + inf_mask, new_state


def policy_mapping_fn(agent_id):
    if agent_id.endswith("0"):
        return "policy_1"  # Choose 1 policy for agent_0
    else:
        # trains against past versions of self
        return np.random.choice(
            ["policy_1", "policy_2", "policy_3", "policy_4"],
            p=[0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3],
        )


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
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=1.0)
    parser.add_argument("--eval-num", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--random-players", type=int, default=0)
    parser.add_argument("--restore", type=str, default="")
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--name", type=str, default="")

    args = parser.parse_args()

    ray.init(
        local_mode=False,
    )

    register_env("yaniv", lambda config: YanivEnv(config))
    ModelCatalog.register_custom_model("TorchRNNModel", TorchRNNModel)

    env = YanivEnv(env_config)
    obs_space = env.observation_space
    act_space = env.action_space

    config = {
        "algorithm": "PPO",
        "env": "yaniv",
        "env_config": env_config,
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
        # "log_level": "INFO",
        "evaluation_num_workers": 0,
        "evaluation_config": {"explore": False},
        "evaluation_interval": args.eval_every,
        "custom_eval_function": make_eval_func(env_config, args.eval_num),
        # hyper params
        "model": {
            "custom_model": "TorchRNNModel",
            "fcnet_hiddens": 64,
            "lstm_cell_size": 64,
            "max_seq_len": 100,
        },
        "batch_mode": "complete_episodes",
        "train_batch_size": 32768,
        "num_sgd_iter": 20,
        "sgd_minibatch_size": 2048,
    }

    resources = PPOTrainer.default_resource_request(config)

    # trainer = PPOTrainer(env="yaniv", config=config)
    # tourny = YanivTournament(env_config, trainers=[trainer])
    # tourny.run(10)
    # print("\n\nRESULTS:\n")
    # tourny.print_stats()
    results = tune.run(
        YanivTrainer,
        resources_per_trial=resources,
        name=args.name,
        config=config,
        stop={"training_iteration": 1000},
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
        max_failures=10,
    )
