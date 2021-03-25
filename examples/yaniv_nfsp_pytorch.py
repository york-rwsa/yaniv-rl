""" An example of learning a Deep-Q Agent on Leduc Holdem
"""
import torch
import os

import rlcard
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament, redirect_to_tqdm
from yaniv_rl.envs import make
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent
from yaniv_rl.utils import ExperimentRunner

from tqdm import trange
from datetime import datetime
import sys

import wandb
default_config = {
    "end_after_n_deck_replacements": 1,
    "end_after_n_steps": 130,
    "early_end_reward": 0,
    "use_scaled_negative_reward": False,
    "max_negative_reward": -1,
    "negative_score_cutoff": 30,
    "seed": 0,
    "feed_both_games": True,
    "feed_both_agents": False
}
default_hyperparams = dict(
    hidden_layers_sizes=[512, 1024, 2048, 1024, 512],
    anticipatory_param=0.5,
    batch_size=256,
    train_every=64,
    rl_learning_rate=0.00005,
    sl_learning_rate=0.00001,
    min_buffer_size_to_learn=10000,
    q_replay_memory_size=int(1e5),
    q_replay_memory_init_size=10000,
    q_train_every=64,
    q_batch_size=256,
    q_mlp_layers=[512, 1024, 2048, 1024, 512],
)
wandb.init(config={**default_config, **default_hyperparams}, project="yaniv_nfsp")

load_model = None
load_model = "/home/jippo/Code/yaniv/yaniv-rl/examples/yaniv_nfsp_pytorch/20210324_20000/model/model_1.pth"
load_scope = "nfsp0"

def main():
    wandb_config = wandb.config
    config = {}
    hyperparams = {}
    for key in wandb_config.keys():
        if key in default_config:
            config[key] = wandb_config[key]
        elif key in default_hyperparams:
            hyperparams[key] = wandb_config[key]

    # Make environment
    env = make("yaniv", config=config)
    eval_env = make("yaniv", config=config)

    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(
            scope="nfsp" + str(i),
            action_num=env.action_num,
            state_shape=env.state_shape,
            device=torch.device("cuda"),
            **hyperparams
        )
        agents.append(agent)
        if load_model is not None:
            state_dict = torch.load(load_model)
            policy_dict = state_dict[load_scope]
            agent.policy_network.load_state_dict(policy_dict)
            q_key = load_scope + '_dqn_q_estimator'
            agent._rl_agent.q_estimator.qnet.load_state_dict(state_dict[q_key])
            target_key = load_scope + '_dqn_target_estimator'
            agent._rl_agent.target_estimator.qnet.load_state_dict(state_dict[target_key])


    rule_agent = YanivNoviceRuleAgent()
    random_agent = RandomAgent(action_num=env.action_num)

    def agent_feed(agent, trajectories):
        for transition in trajectories:
            agent.feed(transition)

    def save_function(agent, model_dir):
        torch.save(agent.get_state_dict(), os.path.join(model_dir, "model_{}.pth".format(i)))
    

    e = ExperimentRunner(
        env,
        eval_env,
        log_every=100,
        save_every=100,
        base_dir="yaniv_nfsp_pytorch",
        config=config,
        training_agent=agents[0],
        vs_agent=rule_agent,
        feed_function=agent_feed,
        save_function=save_function
    )



    e.run_training(
        episode_num=50000,
        eval_every=200,
        eval_vs=[random_agent, rule_agent],
        eval_num=100
    )


if __name__ == "__main__":
    with redirect_to_tqdm():
        main()
