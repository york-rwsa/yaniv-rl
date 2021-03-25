from multiprocessing.context import DefaultContext
from rlcard.agents.random_agent import RandomAgent
import rlcard

from yaniv_rl.envs import make
from yaniv_rl.utils import redirect_to_tqdm
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent
from yaniv_rl.agents import A2CAgentPytorch
from yaniv_rl.utils import ExperimentRunner

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
    "feed_both_agents": False,
    "single_step_actions": True
}

default_hyperparams = {
    "memory_capacity": 10000,
    "max_steps": None,
    "reward_gamma": 0.9323,
    "reward_scale": 1.0,
    "done_penalty": None,
    "actor_hidden_size": 800,
    "critic_hidden_size": 800,
    "critic_loss": "mse",
    "actor_lr": 0.001402,
    "critic_lr": 0.0009512,
    "optimizer_type": "rmsprop",
    "entropy_reg": 0.01626,
    "max_grad_norm": 0.5,
    "batch_size": 139,
    "epsilon_start": 0.9,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.578,
    "train_every_n_episodes": 1,
    "use_cuda": True
}
wandb.init(config={**default_config, **default_hyperparams}, project="yaniv_a2c")

def main():
    wandb_config = wandb.config
    config = {}
    hyperparams = {}
    for key in wandb_config.keys():
        if key in default_config:
            config[key] = wandb_config[key]
        elif key in default_hyperparams:
            hyperparams[key] = wandb_config[key]

    env = make("yaniv", config=config)
    eval_env = make("yaniv", config=config)
    agent = A2CAgentPytorch(
        action_dim=env.action_num, state_shape=env.state_shape, **hyperparams
    )
    rule_agent = YanivNoviceRuleAgent(single_step=config['single_step_actions'])
    random_agent = RandomAgent(action_num=env.action_num)
    
    wandb.watch([agent.actor, agent.critic])

    def agent_feed(agent, trajectories):
        agent.feed_game(trajectories)

    def save_function(agent, model_dir):
        agent.save(model_dir)

    e = ExperimentRunner(
        env,
        eval_env,
        log_every=100,
        save_every=100,
        base_dir="yaniv_a2c_pytorch",
        config=config,
        training_agent=agent,
        vs_agent=agent,
        feed_function=agent_feed,
        save_function=save_function
    )

    e.run_training(
        episode_num=10000,
        eval_every=200,
        eval_vs=[random_agent, rule_agent],
        eval_num=100
    )

if __name__ == "__main__":
    with redirect_to_tqdm():
        main()