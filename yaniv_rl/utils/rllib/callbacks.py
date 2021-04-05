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