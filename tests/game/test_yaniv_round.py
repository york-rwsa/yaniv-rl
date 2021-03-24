import unittest
import numpy as np

from yaniv_rl.game.round import YanivRound
from yaniv_rl.game.player import YanivPlayer
from yaniv_rl.game.card import YanivCard as Card
import yaniv_rl.utils as utils

from itertools import combinations

class TestYanivRound(unittest.TestCase):
    # def setUp(self):
    #     self.round = YanivRound(None, 1, None)

    def _assert_legal_actions(self, actions):
        for action in actions:
            self.assertIn(action, utils.ACTION_LIST)

    def test_legal_pairs(self):
        rnd = YanivRound(None, 1, None)
        player = YanivPlayer(1, None)
        player.hand = [
            Card("S", "T"),
            Card("S", "A"),
            Card("D", "T"),
            Card("S", "J"),
            Card("S", "Q"),
        ]

        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)

        self.assertIn("DTST", actions)
        self.assertIn("STSJSQ", actions)
        self.assertIn("ST", actions)
        self.assertIn("DT", actions)
        self.assertEqual(7, len(actions))

    def test_legal_trips(self):
        rnd = YanivRound(None, 1, None)
        player = YanivPlayer(1, None)
        player.hand = [
            Card("S", "2"),
            Card("D", "2"),
            Card("C", "2"),
        ]

        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)
        expectations = [
            "yaniv",
            "S2",
            "D2",
            "C2",
            "C2D2",
            "C2S2",
            "D2S2",
            "C2S2D2",
            "C2D2S2",
            "D2C2S2",
        ]
        for action in expectations:
            self.assertIn(action, actions)

        self.assertEqual(len(expectations), len(actions))

    def test_legal_quads(self):
        rnd = YanivRound(None, 1, None)
        player = YanivPlayer(1, None)
        player.hand = [
            Card("S", "2"),
            Card("D", "2"),
            Card("H", "2"),
            Card("C", "2"),
        ]

        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)
        expectations = [
            "S2",
            "D2",
            "H2",
            "C2",
            "C2D2",
            "C2H2",
            "C2S2",
            "D2H2",
            "D2S2",
            "H2S2",
            "D2C2H2",
            "C2D2H2",
            "C2H2D2",
            "D2C2S2",
            "C2D2S2",
            "C2S2D2",
            "H2C2S2",
            "C2H2S2",
            "C2S2H2",
            "H2D2S2",
            "D2H2S2",
            "D2S2H2",
            "C2H2S2D2",
            "C2D2S2H2",
            "C2D2H2S2",
            "D2C2S2H2",
            "D2C2H2S2",
            "H2C2D2S2",
        ]
        for action in expectations:
            self.assertIn(action, actions)

        self.assertEqual(len(expectations), len(actions))

    def test_legal_straights(self):
        rnd = YanivRound(None, 1, None)
        player = YanivPlayer(1, None)
        player.hand = [
            Card("S", "9"),
            Card("S", "T"),
            Card("S", "J"),
            Card("S", "Q"),
        ]

        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)
        self.assertIn("S9STSJSQ", actions)
        self.assertIn("STSJSQ", actions)
        self.assertIn("S9STSJ", actions)
        self.assertEqual(7, len(actions))

        player.hand = [
            Card("S", "8"),
            Card("S", "9"),
            Card("S", "T"),
            Card("S", "J"),
            Card("S", "Q"),
        ]

        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)
        self.assertIn("S8S9STSJSQ", actions)
        self.assertIn("S9STSJSQ", actions)
        self.assertIn("S8S9STSJ", actions)
        self.assertIn("STSJSQ", actions)
        self.assertIn("S9STSJ", actions)
        self.assertIn("S8S9ST", actions)
        self.assertEqual(11, len(actions))

        player.hand = [Card("S", "2")]
        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)
        self.assertIn("yaniv", actions)
        self.assertIn("S2", actions)
        self.assertEqual(2, len(actions))

    def test_get_legal_actions_pickup(self):
        rnd = YanivRound(None, 1, None)
        player = YanivPlayer(1, None)
        player.hand = [
            Card("S", "T"),
            Card("S", "A"),
            Card("D", "T"),
            Card("S", "Q"),
        ]
        rnd.discarding = False

        actions = rnd.get_legal_actions([player], 0)
        self._assert_legal_actions(actions)
        self.assertEqual(utils.pickup_actions, actions)

    def test_yaniv_successful_action(self):
        players = [
            YanivPlayer(0, None),
            YanivPlayer(1, None),
            YanivPlayer(2, None),
        ]
        players[0].hand = [
            Card("D", "T"),
        ]
        players[1].hand = [
            Card("H", "2"),
        ]
        players[2].hand = [
            Card("S", "A"),
        ]

        rnd = YanivRound(None, 3, None)
        rnd.current_player = 2
        rnd._perform_yaniv_action(players)

        self.assertEqual(rnd.is_over, True)
        self.assertEqual(rnd.winner, 2)
        self.assertEqual(rnd.scores, [10, 2, 0])

    def test_yaniv_assaf_action(self):
        players = [
            YanivPlayer(0, None),
            YanivPlayer(1, None),
            YanivPlayer(2, None),
        ]
        players[0].hand = [
            Card("S", "A"),
        ]
        players[1].hand = [
            Card("H", "A"),
        ]
        players[2].hand = [
            Card("D", "A"),
        ]

        rnd = YanivRound(None, 3, None)
        rnd.current_player = 2
        rnd._perform_yaniv_action(players)

        self.assertEqual(rnd.is_over, True)
        self.assertEqual(rnd.winner, 1)
        self.assertEqual(rnd.scores, [1, 0, 31])

    def test_yaniv_discard_action(self):
        players = [
            YanivPlayer(0, None),
        ]
        players[0].hand = [
            Card("S", "3"),
            Card("H", "3"),
            Card("D", "5"),
            Card("D", "6"),
            Card("D", "7"),
        ]

        rnd = YanivRound(None, 3, None)
        rnd.current_player = 0
        rnd.known_cards[0] = [players[0].hand[1]]

        rnd._perform_discard_action(players, "S3H3")
        self.assertEqual(utils.cards_to_str(players[0].hand), "D5D6D7")
        self.assertEqual(utils.cards_to_str(rnd.discard_pile[-1]), "H3S3")
        self.assertEqual(rnd.known_cards[0], [])

        rnd._perform_discard_action(players, "D5D6D7")
        self.assertEqual(utils.cards_to_str(players[0].hand), "")
        self.assertEqual(utils.cards_to_str(rnd.discard_pile[-1]), "D5D6D7")

    # just to test - takes too long to run to keep in suite
    # def test_all_legal_actions(self):
    #     deck = utils.init_deck()
    #     players = [
    #         YanivPlayer(0, None),
    #     ]
    #     rnd = YanivRound(None, 1, None)
    #     for handsize in range(6):
    #         for hand in combinations(deck, handsize):
    #             players[0].hand = hand
    #             legal_actions = rnd.get_legal_actions(players, 0)
    #             for a in legal_actions:
    #                 self.assertIn(a, utils.ACTION_LIST)


if __name__ == "__main__":
    unittest.main()
