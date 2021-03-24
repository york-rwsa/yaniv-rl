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
    episode_num = 100000 # mahjong has 100000

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

    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(
            scope="nfsp" + str(i),
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
        agents.append(agent)
    random_agent = RandomAgent(action_num=eval_env.action_num)
    env.set_agents(agents)
    eval_env.set_agents([agents[0], YanivNoviceRuleAgent()])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    logger.log("CONFIG: ")
    logger.log(str(config))
    
    wandb.config.update(config)

    for episode in trange(episode_num, desc="Episodes", file=sys.stdout):
        # First sample a policy for the episode
        for agent in agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for ts in trajectories[i]:
                agents[i].feed(ts)

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
            for i, agent in enumerate(agents):
                torch.save(agent.get_state_dict(), os.path.join(model_dir, "model_{}.pth".format(i)))

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot("NFSP")

    for i, agent in enumerate(agents):
        torch.save(agent.get_state_dict(), os.path.join(model_dir, "model_{}.pth".format(i)))


if __name__ == "__main__":
    with redirect_to_tqdm():
        main()
