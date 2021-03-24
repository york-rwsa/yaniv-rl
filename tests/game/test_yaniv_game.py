import unittest
import numpy as np

from yaniv_rl.game import Round, Game
from yaniv_rl import utils


class TestYanivRound(unittest.TestCase):
    def test_get_player_num(self):
        game = Game()
        player_num = game.get_player_num()
        self.assertEqual(player_num, 2)

        game = Game(3)
        player_num = game.get_player_num()
        self.assertEqual(player_num, 3)

    def test_get_action_num(self):
        game = Game()
        action_num = game.get_action_num()
        self.assertEqual(action_num, 488)

    def test_init_game(self):
        game = Game()
        state, current_player = game.init_game()
        self.assertEqual(current_player, 0)
        for player in game.players:
            self.assertEqual(len(player.hand), utils.INITIAL_NUMBER_OF_CARDS)
        self.assertEqual(len(game.round.discard_pile), 1)
        self.assertEqual(len(game.round.discard_pile[0]), 1)

    def test_step(self):
        game = Game()
        _, current_player = game.init_game()

        # discard first
        action = np.random.choice([a for a in game.get_legal_actions() if a != "yaniv"])
        self.assertNotIn(action, utils.pickup_actions)
        self.assertIn(action, utils.ACTION_LIST)
        _, next_player = game.step(action)
        self.assertEqual(next_player, current_player)

        # then pickup
        action = np.random.choice(game.get_legal_actions())
        self.assertIn(action, utils.pickup_actions)
        _, next_player = game.step(action)
        self.assertEqual(next_player, (current_player + 1) % game.get_player_num())

    def test_pickup_discard(self):
        game = Game()
        _, current_player = game.init_game()

        card = game.players[current_player].hand[0]
        action = str(card)
        _, current_player = game.step(action)

        _, next_player = game.step(utils.DRAW_CARD_ACTION)
        _, next_player = game.step(
            np.random.choice([a for a in game.get_legal_actions() if a != "yaniv"])
        )
        _, current_player = game.step(utils.PICKUP_TOP_DISCARD_ACTION)

        self.assertIn(card, game.players[next_player].hand)

    def test_proceed_game(self):
        game = Game()
        game.init_game()
        while not game.is_over():
            legal_actions = game.get_legal_actions()
            action = np.random.choice(legal_actions)
            self.assertIn(action, utils.ACTION_LIST)
            _, _ = game.step(action)

        if game.round.winner != -1:
            self.assertEqual(game.actions[-1], "yaniv")
        else:
            self.assertEqual(game.actions[-1], "draw_card")

    def test_get_payoffs(self):
        game = Game()
        game.init_game()
        while not game.is_over():
            actions = game.get_legal_actions()
            action = np.random.choice(actions)
            state, _ = game.step(action)

        payoffs = game.get_payoffs()

        for player in game.players:
            player_id = player.get_player_id()
            payoff = payoffs[player_id]

            if game.round.winner == -1:
                self.assertEqual(payoff, -1)
            elif game.round.winner == player_id:
                self.assertEqual(payoff, 1)
            else:
                self.assertEqual(payoff, -(game.round.scores[player_id] / 50))

    def test_end_after_n_deck_replacements(self):
        game = Game()
        game.init_game()
        game._early_end_reward = -1
        game._end_after_n_deck_replacements = 1
        while not game.is_over():
            legal_actions = game.get_legal_actions()
            if utils.DRAW_CARD_ACTION in legal_actions:
                action = utils.DRAW_CARD_ACTION
            else:
                action = np.random.choice(
                    [a for a in legal_actions if a != utils.YANIV_ACTION]
                )

            _, _ = game.step(action)
            self.assertLessEqual(len(game.actions), 84)

        # should take 84 actions
        self.assertEqual(game.round.deck_replacements, 1)
        self.assertEqual(len(game.actions), 84)
        self.assertEqual(game.actions[-1], "draw_card")
        for p in game.get_payoffs():
            self.assertEqual(p, -1)

    def test_end_after_n_steps(self):
        game = Game()
        game.init_game()
        game._early_end_reward = -1
        game._end_after_n_steps = 100
        while not game.is_over():
            legal_actions = game.get_legal_actions()
            if utils.DRAW_CARD_ACTION in legal_actions:
                action = utils.DRAW_CARD_ACTION
            else:
                action = np.random.choice(
                    [a for a in legal_actions if a != utils.YANIV_ACTION]
                )

            _, _ = game.step(action)
            self.assertLessEqual(len(game.actions), 100)

        # should take 84 actions
        self.assertEqual(len(game.actions), 100)
        for p in game.get_payoffs():
            self.assertEqual(p, -1)

if __name__ == "__main__":
    unittest.main()
