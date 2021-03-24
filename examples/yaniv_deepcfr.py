""" An example of learning a DeepCFR Agent on yaniv
"""

import tensorflow as tf
import os

import rlcard
from rlcard.agents import DeepCFR
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger

from yaniv_rl.utils import tournament

config = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 100,
    "early_end_reward": -1,
    "seed": 0,
}
# Make environment
env = rlcard.make("yaniv", config={**config, "allow_step_back": True})
eval_env = rlcard.make("yaniv", config={**config})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 1
evaluate_num = 100
episode_num = 100000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 64

# The paths for saving the logs and learning curves
save_dir = "yaniv_deepcfr"
log_dir = os.path.join(save_dir, "logs/")
model_dir = os.path.join(save_dir, "model/")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Set a global seed
set_global_seed(0)

with tf.Session() as sess:

    # Initialize a global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Set up the agents
    agents = []
    for i in range(env.player_num):
        agent = DeepCFR(sess, scope="deepcfr" + str(i), env=env)
        agents.append(agent)
    random_agent = RandomAgent(action_num=eval_env.action_num)

    env.set_agents(agents)
    eval_env.set_agents([agents[0], random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)
    saver = tf.train.Saver()

    for episode in range(episode_num):
        for agent in agents:
            agent.train()

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
            saver.save(sess, model_dir)

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot("DeepCFR")

    # Save model
    saver.save(sess, model_dir)
