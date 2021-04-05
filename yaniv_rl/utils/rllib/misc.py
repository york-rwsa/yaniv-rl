import numpy as np
from ray.tune.logger import pretty_print

from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

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

        agent = trainer
        rule_agent = YanivNoviceRuleAgent(
            single_step=env_config.get("single_step", True)
        )
        agent_id = "player_0"
        rules_id = "player_1"

        env = YanivEnv(env_config)

        wins = 0
        draws = 0
        assafs = 0
        total_steps = 0
        scores = [[], []]
        for _ in range(eval_num):
            done = {"__all__": False}
            obs = env.reset()

            steps = 0
            while not done["__all__"]:
                if env.current_player == 0:
                    action = agent.compute_action(obs[agent_id], policy_id="policy_1")
                    obs, reward, done, info = env.step({agent_id: action})
                else:
                    state = env.game.get_state(1)
                    extracted_state = {}
                    extracted_state["raw_obs"] = state
                    extracted_state["raw_legal_actions"] = [
                        a for a in state["legal_actions"]
                    ]

                    action = rule_agent.step(extracted_state)
                    obs, reward, done, info = env.step(
                        {rules_id: action}, raw_action=True
                    )

                steps += 1

            # print(episode_reward, steps, reward)

            # metrics
            if reward[agent_id] == 0:
                draws += 1
            elif reward[agent_id] > 0:
                wins += 1
            total_steps += steps

            # assaf contains the player id that assafed, or None
            if env.game.round.assaf == 0:
                assafs += 1

            s = env.game.round.scores
            if s is not None:
                if s[0] > 0:
                    scores[0].append(env.game.round.scores[0])
                if s[1] > 0:
                    scores[1].append(env.game.round.scores[1])

        eval_vs = "eval_rules_"
        metrics = {
            eval_vs + "draw_rate": draws / eval_num,
            eval_vs + "avg_roundlen": total_steps / eval_num,
            eval_vs + "win_rate": wins / eval_num,
            eval_vs + "assaf_rate": assafs / eval_num,
            eval_vs + "self_avg_losing_score": np.mean(scores[0])
            if len(scores[0]) > 0
            else 0,
            eval_vs + "oppt_avg_losing_score": np.mean(scores[1])
            if len(scores[1]) > 0
            else 0,
        }

        print(pretty_print(metrics), "\n\n\n")

        return metrics

    return yaniv_eval