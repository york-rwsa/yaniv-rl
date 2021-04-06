import numpy as np
from ray.tune.logger import pretty_print

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

from .tournament import YanivTournament


def copy_weights(to_policy, from_policy, trainer):
    """copy weights from from_policy to to_policy without changing from_policy"""
    temp_weights = {}  # temp storage with to_policy keys & from_policy values
    for (k, v), (k2, v2) in zip(
        trainer.get_policy(to_policy).get_weights().items(),
        trainer.get_policy(from_policy).get_weights().items(),
    ):
        temp_weights[k] = v2

    # set weights
    trainer.set_weights(
        {
            to_policy: temp_weights,  # weights or values from from_policy with to_policy keys
        }
    )

    # To check
    for (k, v), (k2, v2) in zip(
        trainer.get_policy(to_policy).get_weights().items(),
        trainer.get_policy(from_policy).get_weights().items(),
    ):
        assert (v == v2).all()

    print("{} == {}".format(to_policy, from_policy))


def shift_policies(trainer, new, p2, p3, p4):
    copy_weights(p4, p3, trainer)
    copy_weights(p3, p2, trainer)
    copy_weights(p2, new, trainer)


def make_eval_func(env_config, eval_num):
    def yaniv_eval(trainer, eval_workers):
        print("\n\n\n************** EVALUATION **************")

        t = YanivTournament(env_config, [trainer])
        res = t.run(eval_num)
        print(pretty_print(res), "\n\n\n")

        eval_vs = "eval_rules_"
        metrics = {
            eval_vs + "draw_rate": res["game"]["avg_draws"],
            eval_vs + "avg_roundlen": res["game"]["avg_roundlen"],
            eval_vs + "win_rate": res["player"]["player_0"]["avg_wins"],
            eval_vs + "assaf_rate": res["player"]["player_0"]["avg_assafs"],
            eval_vs
            + "self_avg_losing_score": res["player"]["player_0"]["avg_losing_score"],
            eval_vs
            + "oppt_avg_losing_score": np.mean(
                [
                    val["avg_losing_score"]
                    for key, val in res["player"].items()
                    if key != "player_0"
                ]
            ),            
            eval_vs
            + "oppt_assaf_rate": np.mean(
                [
                    val["avg_assafs"]
                    for key, val in res["player"].items()
                    if key != "player_0"
                ]
            ),

        }

        return metrics

    return yaniv_eval