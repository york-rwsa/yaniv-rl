""" An example of learning a Deep-Q Agent on yaniv
"""
import torch
import os
import sys

import rlcard
from rlcard.agents import A2CAgentPytorch as A2CAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament, redirect_to_tqdm, pickup_actions
import yaniv_rl.utils as utils
from datetime import datetime
from tqdm import trange
yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

import wandb


def reward_func(state, next_state, action):
    if action not in utils.pickup_actions and len(action) > 2:
        return 0.01 * len(action)
    else:
        return -0.05


load_model = None
# load_model = 'yaniv_a2c_pytorch/20210322_800units/model'


def main():
    wandb.init(project="yaniv_a2c")
    config = {
        "end_after_n_deck_replacements": 1,
        "end_after_n_steps": 75,
        "early_end_reward": -1,
        "use_scaled_negative_reward": False,
        "max_negative_reward": -1,
        "negative_score_cutoff": 30,
        "seed": 0,
        "feed_both_games": False,
    }

    hyperparams = {
        "memory_capacity": 10000,
        "max_steps": None,
        "roll_out_n_steps": 10,
        "reward_gamma": 0.9,
        "reward_scale": 1.0,
        "done_penalty": None,
        "actor_hidden_size": 800,
        "critic_hidden_size": 800,
        "critic_loss": "mse",
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "optimizer_type": "adam",
        "entropy_reg": 0.01,
        "max_grad_norm": 0.5,
        "batch_size": 100,
        "epsilon_start": 0.9,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99,
        "train_every_n_episodes": 1,
    }

    # Make environment
    env = rlcard.make("yaniv", config=config)
    eval_env = rlcard.make("yaniv", config=config)

    env.reward_func = reward_func
    eval_env.reward_func = reward_func

    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = 500
    evaluate_num = 500  # mahjong has 1000
    episode_num = 1500
    save_every = 1000
    log_training_stats_every = 100

    # The paths for saving the logs and learning curves
    save_dir = "yaniv_a2c_pytorch/{}".format(datetime.now().strftime("%Y%m%d"))
    log_dir = os.path.join(save_dir, "logs/")
    model_dir = os.path.join(save_dir, "model/")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Set a global seed
    set_global_seed(0)

    agent = A2CAgent(
        action_dim=env.action_num, state_shape=env.state_shape, **hyperparams
    )
    random_agent = RandomAgent(action_num=eval_env.action_num)
    # env.set_agents([agent, agent])
    training_agents = [agent, agent]
    env.set_agents(training_agents)
    # eval_env.set_agents([agent, YanivNoviceRuleAgent()])
    eval_env.set_agents([agent, random_agent])

    wandb.config.update({"vs": "self"})
    wandb.config.update({"eval_vs": "random"})

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    logger.log("CONFIG: ")
    logger.log(str(config))

    wandb.config.update(config)
    wandb.config.update(hyperparams)
    wandb.watch([agent.actor, agent.critic])

    if load_model is not None:
        agent.load(load_model)

    statlogger = YanivStatLogger(logger)
    for episode in trange(episode_num, desc="Episodes", file=sys.stdout):
        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        statlogger.add_game(trajectories, env, 0)
        # Feed transitions into agent memory, and train the agent
        # for ts in trajectories[0]:
        #     agent.feed(ts)
        # agent.feed_game(trajectories[0])

        if config.get("feed_both_games"):
            if training_agents[1].use_raw:
                agent.feed_game(
                    list(
                        map(
                            lambda t: [t[0], utils.ACTION_SPACE[t[1]], *t[2:]],
                            trajectories[1],
                        )
                    )
                )
            else:
                agent.feed_game(trajectories[1])

        if episode != 0 and episode % log_training_stats_every == 0:
            statlogger.log_stats()

        if episode != 0 and episode % evaluate_every == 0:
            r = tournament(eval_env, evaluate_num)

            wandb.log(
                {
                    "eval_payoff": r["payoffs"][0],
                    "eval_draws": r["draws"],
                    "eval_roundlen": r["roundlen"],
                    "eval_assafs": r["assafs"][0],
                    "eval_win_rate": r["wins"][0] / evaluate_num,
                },
            )

            logger.log("\n\n########## Evaluation {} ##########".format(episode))
            logger.log(
                "Timestep: {}, avg roundlen: {}".format(env.timestep, r["roundlen"])
            )
            for i in range(env.player_num):
                logger.log(
                    "Agent {}:\nWins: {}, Draws: {}, Assafs: {}, Payoff: {}".format(
                        i,
                        r["wins"][i],
                        r["draws"],
                        r["assafs"][i],
                        r["payoffs"][i],
                    )
                )

            logger.log_performance(env.timestep, r["payoffs"][0])

        if episode != 0 and episode % save_every == 0:
            agent.save(model_dir)

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot("DQN")

    # Save model
    agent.save(model_dir)


class YanivStatLogger:
    def __init__(self, logger):
        self.avg_stats = {}
        self.count_stats = {}
        self.count = 0
        self.reset_stats()
        self.logger = logger

    def reset_stats(self):
        self.avg_stats = {"roundlen": 0, "avg_reward": 0, "pos_rewards": 0}
        self.count_stats = {"draws": 0}
        self.count = 0

    def add_game(self, trajectories, env, player_id):
        self.avg_stats["roundlen"] += len(env.game.actions)

        rewards = [t[2] for t in trajectories[player_id]]
        self.avg_stats["avg_reward"] += sum(rewards) / len(rewards)

        pos_rewards = [r for r in rewards if r > 0 and r != 1]
        self.avg_stats["pos_rewards"] += len(pos_rewards)

        if env.game.round.winner == -1:
            self.count_stats["draws"] += 1

        self.count += 1

    def calc_average(self):
        for key in self.avg_stats.keys():
            self.avg_stats[key] /= self.count

    def log_stats(self):
        self.calc_average()
        stats = {**self.count_stats, **self.avg_stats}
        self.logger.log("{}".format(stats))
        wandb.log(stats)
        self.reset_stats()


if __name__ == "__main__":
    with redirect_to_tqdm():
        main()