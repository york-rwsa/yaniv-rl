import unittest
import numpy as np

import rlcard
from yaniv_rl.envs import make
from rlcard.agents.random_agent import RandomAgent
# from determism_util import is_deterministic

class TestYanivEnv(unittest.TestCase):
    def test_reset_and_extract_state(self):
        env = make("yaniv")
        state, _ = env.reset()
        self.assertEqual(state["obs"].size, 6 * 52)

    # def test_is_deterministic(self):
    #     self.assertTrue(is_deterministic("yaniv"))

    def test_get_legal_actions(self):
        env = make("yaniv")
        env.set_agents([RandomAgent(env.action_num) for _ in range(env.player_num)])
        env.reset()
        legal_actions = env._get_legal_actions()
        for legal_action in legal_actions:
            self.assertLessEqual(legal_action, env.action_num - 1)

    def test_step(self):
        env = make("yaniv")
        state, _ = env.reset()
        action = np.random.choice(state["legal_actions"])
        _, player_id = env.step(action)
        current_player_id = env.game.round.current_player
        self.assertEqual(player_id, current_player_id)

    def test_run(self):
        env = make("yaniv")
        env.set_agents([RandomAgent(env.action_num) for _ in range(env.player_num)])
        trajectories, payoffs = env.run(is_training=False)
        self.assertEqual(len(trajectories), 2)
        for payoff in payoffs:
            self.assertLessEqual(-1, payoff)
            self.assertLessEqual(payoff, 1)
        trajectories, payoffs = env.run(is_training=True)
        for payoff in payoffs:
            self.assertLessEqual(-1, payoff)
            self.assertLessEqual(payoff, 1)


if __name__ == "__main__":
    unittest.main()
