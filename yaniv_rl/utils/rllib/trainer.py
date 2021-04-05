import ray

from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray import tune

from . import shift_policies


class YanivTrainer(tune.Trainable):
    def setup(self, config):
        self.trainer = PPOTrainer(env="yaniv", config=config)
        self.config = config

    def step(self):
        result = self.trainer.train()

        if result["custom_metrics"]["win_mean"] > 0.52:
            shift_policies(self.trainer, "policy_1", "policy_2", "policy_3", "policy_4")
            print("weights shifted")
            weights = ray.put(self.trainer.workers.local_worker().save())
            self.trainer.workers.foreach_worker(lambda w: w.restore(ray.get(weights)))
            print("weights synced")

        return result

    def save_checkpoint(self, dir):
        return self.trainer.save_checkpoint(dir)

    def load_checkpoint(self, checkpoint):
        self.trainer.load_checkpoint(checkpoint)