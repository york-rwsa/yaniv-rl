from yaniv_rl.utils.utils import DISCARD_ACTION_LIST
from yaniv_rl import utils

import unittest


class TestYanivEnv(unittest.TestCase):
    def test_joined_action_space(self):
        self.assertEqual(
            len(utils.JOINED_ACTION_SPACE),
            1 + len(utils.DISCARD_ACTION_LIST) * len(utils.pickup_actions),
        )
        # print(utils.JOINED_ACTION_SPACE)


if __name__ == "__main__":
    unittest.main()
