import pickle5 as pickle
import os
import ray

from ray.rllib.agents.registry import get_trainer_class
from ray import tune

from . import shift_policies


class YanivTrainer(tune.Trainable):
    def setup(self, config):
        algo = config.pop("algorithm")
        eval_weights = config.pop("evaluation_weights", None)

        self.export_model_every = config.pop("export_model_every", 10)
        self.update_winrate = config.pop("update_self_play_param_win_rate", 0.5)
        
        self.trainer = get_trainer_class(algo)(env="yaniv", config=config)

        if eval_weights is not None:
            self.trainer.set_weights({
                "eval_policy": eval_weights
            })

        self.config = config

    def step(self):
        result = self.trainer.train()

        if result["custom_metrics"]["win_mean"] > self.update_winrate:
            shift_policies(self.trainer, "policy_1", "policy_2", "policy_3", "policy_4")
            print("weights shifted")
            weights = ray.put(self.trainer.workers.local_worker().save())
            self.trainer.workers.foreach_worker(lambda w: w.restore(ray.get(weights)))
            print("weights synced")


        if self.export_model_every is not None and self.iteration % self.export_model_every == 0:
            # save model
            path = os.path.join(self.logdir, 'models/model-{:06d}.pkl'.format(self.iteration))
            state = self.trainer.get_policy("policy_1").get_state()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(state, f)

            print("pickled state to", path)

        return result

    def save_checkpoint(self, dir):
        return self.trainer.save_checkpoint(dir)

    def load_checkpoint(self, checkpoint):
        self.trainer.load_checkpoint(checkpoint)
