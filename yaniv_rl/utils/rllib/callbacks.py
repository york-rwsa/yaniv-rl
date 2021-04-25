from yaniv_rl import utils
from ray.rllib.agents.callbacks import DefaultCallbacks

class YanivCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        pass

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        """
        Used in order to add custom metrics to our tensorboard data
        """
        # Get env refernce from rllib wraper
        # env = base_env.get_unwrapped()[0]

        final_rewards = {k: r[-1] for k, r in episode._agent_reward_history.items()}

        episode.custom_metrics["final_reward"] = final_rewards["player_0"]
        episode.custom_metrics["win"] = 1 if final_rewards["player_0"] > 0 else 0
        episode.custom_metrics["draw"] = 1 if final_rewards["player_0"] == 0 else 0
        if final_rewards["player_0"] < 0:
            episode.custom_metrics["negative_reward"] = final_rewards["player_0"]


        # Get env refernce from rllib wraper
        env = base_env.get_unwrapped()[0]

        metrics = {}
        for pid in env._get_players():
            metrics.update({"draw": 0, pid + "_win": 0, pid + "_assaf": 0})

        winner = env.game.round.winner
        if winner == -1:
            metrics["draw"] = 1
        else:
            winner_id = env._get_player_string(winner)
            metrics[winner_id + "_win"] = 1
            metrics[winner_id + "_winning_hands"] = utils.get_hand_score(
                env.game.players[winner].hand
            )

        assaf = env.game.round.assaf

        if assaf is not None:
            metrics[env._get_player_string(assaf) + "_assaf"] = 1

        s = env.game.round.scores
        if s is not None:
            for i in range(env.num_players):
                if s[i] > 0:
                    metrics[env._get_player_string(i) + "_losing_score"] = s[i]

        episode.custom_metrics.update(metrics)

    def on_sample_end(self, worker, samples, **kwargs):
        pass

    def on_train_result(self, trainer, result, **kwargs):
        pass

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        pass