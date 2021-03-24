import tensorflow as tf
from keras import backend as K

import rlcard

from rlcard.agents import A2CAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament

import os

def runModel():
    config = {
        "end_after_n_deck_replacements": 2,
        "end_after_n_steps": 0,
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
    episode_num = 10000
    save_every = 1000

    # Set a global seed
    set_global_seed(0)
    save_dir = "yaniv_a2c"
    log_dir = os.path.join(save_dir, "logs/")
    model_dir = os.path.join(save_dir, "model/")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    a2cagent = A2CAgent(
        action_num=eval_env.action_num,
        state_shape=env.state_shape,
    )
    random_agent = RandomAgent(action_num=eval_env.action_num)
    agents = [a2cagent, random_agent]
    env.set_agents(agents)
    eval_env.set_agents(agents)

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    

    for episode in range(episode_num):
        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for ts in trajectories[0]:
            agents[0].feed(ts)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            payoffs, wins, draws, roundlen = tournament(eval_env, evaluate_num)

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
            agents[0].save_weights(model_dir)

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot("A2C")

    # Save model
    #saver = tf.train.Saver()          
    agents[0].save_weights(model_dir)


if __name__ == "__main__":
    runModel()