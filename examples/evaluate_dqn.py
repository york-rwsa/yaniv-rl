import click
import tensorflow as tf
import matplotlib.pyplot as plt
import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament

plt.style.use("ggplot")


# @click.command()
# @click.option('--path', required=True, help='Path to model')
# @click.option('--num', default=400, help='Number of iterations')
# @click.option('--position', default=0, help='Player position')
# @click.option('--opponent', default='random', help='Opponent strategy (random or deep_cfr)')
def run(path: str, num: int, position: int, opponent: str):
    # Set a global seed
    # set_global_seed(123)
    env = rlcard.make("yaniv", config={"seed": 0, "end_after_n_steps": 0})

    agents = []
    for _ in range(env.player_num):
        agent = RandomAgent(action_num=env.action_num)
        agents.append(agent)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    with graph.as_default():
        agent = DQNAgent(
            sess,
            scope="dqn",
            action_num=env.action_num,
            replay_memory_size=20000,
            replay_memory_init_size=1000,
            train_every=0,
            state_shape=env.state_shape,
            mlp_layers=[512, 512],
        )
        if opponent == "self":
            agents = [agent for _ in agents]
        else:
            agents[position] = agent

    with sess.as_default():
        with graph.as_default():
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(path))

    env.set_agents(agents)
    payoffs, wins, draws, roundlen = tournament(env, num)
    print(payoffs, wins, draws, roundlen)


if __name__ == "__main__":
    # run()
    run("examples/yaniv/2021-02-14_dqn", 20, 1, "random")
