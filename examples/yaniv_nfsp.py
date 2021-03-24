"""
    File name: rlcard.examples.gin_rummy_nfsp.py
    Author: William Hale
    Date created: 2/12/2020

    An example of learning a NFSP Agent on GinRummy
"""

import tensorflow as tf
import os

import rlcard

from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament


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
episode_num = 10000
save_every = 1000

# The initial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 64

# Set a global seed
set_global_seed(0)
save_dir = "yaniv_nfsp_tensorflow"
log_dir = os.path.join(save_dir, "logs/")
model_dir = os.path.join(save_dir, "model/")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

with tf.Session() as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Set up the agents
    agents = []
    for i in range(env.player_num):
        agent = NFSPAgent(
            sess,
            scope="nfsp" + str(i),
            action_num=env.action_num,
            state_shape=env.state_shape,
            hidden_layers_sizes=[512, 1024, 2048, 1024, 512],
            anticipatory_param=0.5,
            batch_size=256,
            rl_learning_rate=0.00005,
            sl_learning_rate=0.00001,
            min_buffer_size_to_learn=memory_init_size,
            q_replay_memory_size=int(1e5),
            q_replay_memory_init_size=memory_init_size,
            train_every=train_every,
            q_train_every=train_every,
            q_batch_size=256,
            q_mlp_layers=[512, 1024, 2048, 1024, 512],
        )
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
        # First sample a policy for the episode
        for agent in agents:
            agent.sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for i in range(env.player_num):
            for ts in trajectories[i]:
                agents[i].feed(ts)

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
    logger.plot("NFSP")

    # Save model
    #saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, "model"))
