from datetime import datetime

from yaniv_rl import utils
from yaniv_rl.utils import tournament
import sys
import os
from rlcard.utils.logger import Logger
from tqdm import trange
import wandb


class ExperimentRunner:
    def __init__(
        self,
        env,
        eval_env,
        log_every,
        save_every,
        base_dir,
        config,
        training_agent,
        vs_agent,
        feed_function,
        save_function
    ):
        self.save_dir = "{}/{}".format(base_dir, datetime.now().strftime("%Y%m%d"))
        self.log_dir = os.path.join(self.save_dir, "logs/")
        self.model_dir = os.path.join(self.save_dir, "model/")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.log_every = log_every
        self.save_every = save_every

        self.config = config
        self.env = env
        self.eval_env = eval_env
        self.agent = training_agent
        self.training_agents = [self.agent, vs_agent]
        self.env.set_agents(self.training_agents)

        self.logger = Logger(self.log_dir)
        self.logger.log("CONFIG: ")
        self.logger.log(str(config))
        self.stat_logger = YanivStatLogger(self.logger)

        self.feed_function = feed_function
        self.save_function = save_function

        self.action_space = utils.JOINED_ACTION_SPACE if config['single_step_actions'] else utils.ACTION_SPACE

    def feed_game(self, agent, trajectories, player_id):
        self.feed_function(agent, trajectories[player_id])
        
        if self.config.get("feed_both_games"):
            other_traj = trajectories[player_id + 1 % len(self.training_agents)]
            if self.training_agents[player_id + 1 % len(self.training_agents)].use_raw:
                self.feed_function(agent, 
                    list(
                        map(
                            lambda t: [t[0], self.action_space[t[1]], *t[2:]],
                            other_traj,
                        )
                    )
                )
            else:
                self.feed_function(agent, other_traj)

    def run_training(self, episode_num, eval_every, eval_vs, eval_num):
        for episode in trange(episode_num, desc="Episodes", file=sys.stdout):
            # Generate data from the environment
            trajectories, _ = self.env.run(is_training=True)
            self.stat_logger.add_game(trajectories, self.env, 0)

            self.feed_game(self.agent, trajectories, 0)
            if self.config['feed_both_agents']:
                self.feed_game(self.training_agents[1], trajectories, 1)
            
            if episode != 0 and episode % self.log_every == 0:
                self.stat_logger.log_stats()

            if episode != 0 and episode % self.save_every == 0:
                self.save_function(self.agent, self.model_dir)

            if episode != 0 and episode % eval_every == 0:
                self.logger.log("\n\n########## Evaluation {} ##########".format(episode))
                self.evaluate_perf(eval_vs, eval_num)
            
        self.evaluate_perf(eval_vs, eval_num)
        self.save_function(self.agent, self.model_dir)

    def evaluate_perf(self, eval_vs, eval_num):
        if isinstance(eval_vs, list):
            for vs in eval_vs:
                self.run_evaluation(vs, eval_num)
        else:
            self.run_evaluation(eval_vs, eval_num)

    def run_evaluation(self, vs, num):
        self.eval_env.set_agents([self.agent, vs])
        self.logger.log("eval vs {}".format(vs.__class__.__name__))
        r = tournament(self.eval_env, num)
        
        eval_vs = "eval_{}_".format(vs.__class__.__name__)
        wandb.log(
            {
                eval_vs + "payoff": r["payoffs"][0],
                eval_vs + "draws": r["draws"],
                eval_vs + "roundlen": r["roundlen"],
                eval_vs + "assafs": r["assafs"][0],
                eval_vs + "win_rate": r["wins"][0] / num,
            },
        )

        self.logger.log(
            "Timestep: {}, avg roundlen: {}".format(self.env.timestep, r["roundlen"])
        )
        for i in range(self.env.player_num):
            self.logger.log(
                "Agent {}:\nWins: {}, Draws: {}, Assafs: {}, Payoff: {}".format(
                    i,
                    r["wins"][i],
                    r["draws"],
                    r["assafs"][i],
                    r["payoffs"][i],
                )
            )

        self.logger.log_performance(self.env.timestep, r["payoffs"][0])


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
