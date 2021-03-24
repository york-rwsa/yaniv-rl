""" An example of learning a Deep-Q Agent on Leduc Holdem
"""
from random import random
import torch
import os

import rlcard
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament, redirect_to_tqdm
yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

from tqdm import trange
from datetime import datetime
import sys

import wandb


def main():
    wandb.init(project="yaniv")

    config = {
        "end_after_n_deck_replacements": 0,
        "end_after_n_steps": 500,
        "early_end_reward": 0,
        "use_scaled_negative_reward": False,
        "seed": 0,
    }

    # Make environment
    env = rlcard.make("yaniv", config=config)
    eval_env = rlcard.make("yaniv", config=config)

    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = 1000
    evaluate_num = 100  # mahjong has 1000
    save_every = 1000
    episode_num = 10000  # mahjong has 100000

    # The initial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 64

    # Set a global seed
    set_global_seed(0)
    save_dir = "yaniv_nfsp/{}".format(datetime.now().strftime("%Y%m%d_%H%M"))
    log_dir = os.path.join(save_dir, "logs/")
    model_dir = os.path.join(save_dir, "model/")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    nfsp_agent = NFSPAgent(
        scope="nfsp",
        action_num=env.action_num,
        state_shape=env.state_shape,
        hidden_layers_sizes=[512, 1024, 2048, 1024, 512],
        anticipatory_param=0.5,
        batch_size=256,
        train_every=train_every,
        rl_learning_rate=0.00005,
        sl_learning_rate=0.00001,
        min_buffer_size_to_learn=memory_init_size,
        q_replay_memory_size=int(1e5),
        q_replay_memory_init_size=memory_init_size,
        q_train_every=train_every,
        q_batch_size=256,
        q_mlp_layers=[512, 1024, 2048, 1024, 512],
        device=torch.device("cuda"),
    )

    agents = [nfsp_agent, YanivNoviceRuleAgent()]
    env.set_agents(agents)
    eval_env.set_agents(agents)

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    logger.log("CONFIG: ")
    logger.log(str(config))

    wandb.config.update(config)

    for episode in trange(episode_num, desc="Episodes", file=sys.stdout):
        agents[0].sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agents[0].feed(ts)
        trajlen = len(trajectories[0])
        rewards = [t[2] for t in trajectories[0]]
        positive_rewards = [r for r in rewards if r > 0]
        lenposrewards = len(positive_rewards)
        avgposreward = sum(positive_rewards) / lenposrewards if lenposrewards > 0 else 0
        wandb.log(
            {
                "roundlen": trajlen,
                "avg_reward": sum(rewards) / trajlen,
                "number_of_pos_rewards": lenposrewards,
                "avg_pos_reward": avgposreward,
            }
        )

        if episode % evaluate_every == 0:
            payoffs, wins, draws, roundlen = tournament(eval_env, evaluate_num)
            wandb.log(
                {
                    "eval_payoff": payoffs[0],
                    "eval_wins": wins,
                    "eval_draws": draws,
                    "eval_roundlen": roundlen,
                }
            )

            logger.log("\n\n########## Evaluation {} ##########".format(episode))
            logger.log("Timestep: {}, avg roundlen: {}".format(env.timestep, roundlen))
            for i in range(env.player_num):
                logger.log(
                    "Agent {}:\nWins: {}, Draws: {}, Payoff: {}".format(
                        i, wins[i], draws, payoffs[i]
                    )
                )

            logger.log_performance(env.timestep, payoffs[0])

        if episode % save_every == 0:
            torch.save(agents[0].get_state_dict(), os.path.join(model_dir, "model.pth"))

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot("NFSP")

    torch.save(agents[0].get_state_dict(), os.path.join(model_dir, "model.pth"))


if __name__ == "__main__":
    with redirect_to_tqdm():
        main()
