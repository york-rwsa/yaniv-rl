""" An example of learning a Deep-Q Agent on Yanivcd e
"""

import tensorflow as tf
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament


def main():
    config = {
        "end_after_n_deck_replacements": 0,
        "end_after_n_steps": 0,
        "early_end_reward": -1,
        "seed": 0,
        "env_num": 1,
    }
    # Make environment
    env = rlcard.make("yaniv", config=config)
    eval_env = rlcard.make("yaniv", config=config)

    # Set the iterations numbers and how frequently we evaluate the performance
    evaluate_every = 100
    evaluate_num = 100
    episode_num = 1000

    # The intial memory size
    memory_init_size = 1000

    # Train the agent every X steps
    train_every = 1
    save_every = 1

    # Set a global seed
    set_global_seed(0)
    save_dir = "yaniv_dqn"
    log_dir = os.path.join(save_dir, "logs/")
    model_dir = os.path.join(save_dir, "model/")

    with tf.Session() as sess:

        # Initialize a global step
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Set up the agents
        agent = DQNAgent(
            sess,
            scope="dqn",
            action_num=env.action_num,
            replay_memory_size=20000,
            replay_memory_init_size=memory_init_size,
            train_every=train_every,
            state_shape=env.state_shape,
            mlp_layers=[512, 512],
        )
        random_agent = RandomAgent(action_num=eval_env.action_num)
        env.set_agents([agent, random_agent])
        eval_env.set_agents([agent, random_agent])

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Init a Logger to plot the learning curve
        logger = Logger(log_dir)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % evaluate_every == 0:
                payoffs, wins, draws, roundlen = tournament(eval_env, evaluate_num)

                logger.log("\n\n########## Evaluation {} ##########".format(episode))
                logger.log(
                    "Timestep: {}, avg roundlen: {}".format(env.timestep, roundlen)
                )
                for i in range(env.player_num):
                    logger.log(
                        "Agent {}:\nWins: {}, Draws: {}, Payoff: {}".format(
                            i, wins[i], draws, payoffs[i]
                        )
                    )

                logger.log_performance(env.timestep, payoffs[0])

            # Save model
            if episode % save_every == 0:
                saver.save(sess, os.path.join(save_dir, "model"))

        # Close files in the logger
        logger.close_files()

        # Plot the learning curve
        logger.plot("DQN")

        # Save model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(save_dir, "model"))


if __name__ == "__main__":
    main()