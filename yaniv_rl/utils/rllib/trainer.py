import ray

from ray.rllib.agents.registry import get_trainer_class
from ray import tune

from . import shift_policies


class YanivTrainer(tune.Trainable):
    def setup(self, config):
        algo = config.pop("algorithm")
        checkpoint = config.pop("evaluation_checkpoint")

        self.trainer = get_trainer_class(algo)(env="yaniv", config=config)

        if checkpoint is not None:
            loader = get_trainer_class(algo)(env="yaniv", config=config)
            loader.load_checkpoint(checkpoint)
            policy = loader.get_policy("policy_1").get_weights()
            self.trainer.set_weights({
                "eval_policy": policy
            })

        self.config = config

    def step(self):
        result = self.trainer.train()

        if result["custom_metrics"]["win_mean"] > 0.475:
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
