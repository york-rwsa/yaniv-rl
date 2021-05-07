from copy import deepcopy
from itertools import product
import random
import numpy as np
from numpy.lib.arraysetops import isin

from yaniv_rl.game.dealer import YanivDealer
from yaniv_rl.game.player import YanivPlayer
from yaniv_rl.game.round import YanivRound
from yaniv_rl import utils


class YanivGame(object):
    def __init__(self, num_players=2, single_step_actions=False, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.payoffs = [0 for _ in range(self.num_players)]

        # default config
        self._end_after_n_deck_replacements = 0
        self._end_after_n_steps = 0
        self._early_end_reward = 0
        self._use_scaled_negative_reward = False
        self._use_scaled_positive_reward = False
        self._max_negative_reward = -1
        self._negative_score_cutoff = 50
        self._starting_player = "random"
        self._starting_hands = {}

        self._single_step_actions = single_step_actions

    def init_game(self):
        """Initialize players and state

        Returns:
            (tuple): Tuple containing:
                (dict): The first state in one game
                (int): Current player's id
        """
        # Initalize payoffs
        self.payoffs = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = YanivDealer(self.np_random)
        # Initialize four players to play the game
        self.players = [YanivPlayer(i, self.np_random) for i in range(self.num_players)]

        # deal with predefined starting hands
        for pid, starting_hand in self._starting_hands.items():
            player = self.players[pid]

            if isinstance(starting_hand, list):
                starting_hand = random.choices(starting_hand, k=1)[0]

            assert len(starting_hand) == 10
            
            cards = list(filter(lambda c: str(c) in starting_hand, self.dealer.deck))
            for card in cards:
                player.hand.append(card)
                self.dealer.deck.remove(card)
            assert len(player.hand) == 5
    
        # Deal 5 cards to each player
        for player in self.players:
            if player.player_id in self._starting_hands.keys():
                continue

            for _ in range(utils.INITIAL_NUMBER_OF_CARDS):
                player.hand.append(self.dealer.draw_card())

        for player in self.players:
            player.save_starting_hand()

        # Initialize a Round
        self.round = YanivRound(
            self.dealer,
            self.num_players,
            self.np_random,
            starting_player=self.np_random.randint(0, self.num_players)
            if self._starting_player == "random"
            else self._starting_player,
        )
        self.round.flip_top_card()

        # Save the hisory for stepping back to the last state.
        self.history = []
        self.actions = []

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def configure(self, config):
        """Specifiy some game specific parameters, such as player number"""
        self._end_after_n_deck_replacements = config["end_after_n_deck_replacements"]
        self._end_after_n_steps = config["end_after_n_steps"]
        self._early_end_reward = config["early_end_reward"]
        self._use_scaled_negative_reward = config["use_scaled_negative_reward"]
        self._use_scaled_positive_reward = config["use_scaled_positive_reward"]
        self._max_negative_reward = config["max_negative_reward"]
        self._negative_score_cutoff = config["negative_score_cutoff"]
        self._starting_player = config["starting_player"]
        self._starting_hands = config["starting_hands"]

    def step(self, action):
        """Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        """

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_round))

        self.players[self.round.current_player].actions.append(action)
        self.actions.append(action)
        if self._end_after_n_steps > 0 and len(self.actions) >= self._end_after_n_steps:
            self.round.winner = -1
            self.round.is_over = True

        if not self._single_step_actions or action == utils.YANIV_ACTION:
            return self._step(action)

        assert isinstance(
            action, tuple
        ), "in single step actions the action must be a tuple"
        cur_player = self.round.current_player
        _, next_player = self._step(action[0])
        assert cur_player == next_player

        return self._step(action[1])

    def _step(self, action):
        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)

        # end the game if repalce deck is required with everyone losing
        if (
            self._end_after_n_deck_replacements > 0
            and self.round.deck_replacements >= self._end_after_n_deck_replacements
        ):
            self.round.winner = -1
            self.round.is_over = True

        return state, player_id

    def step_back(self):
        """Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        """
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_state(self, player_id):
        """Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        """
        state = self.round.get_state(self.players, player_id)
        state["player_num"] = self.get_player_num()
        state["current_player"] = self.round.current_player
        state["legal_actions"] = self._get_legal_actions(player_id)
        return state

    def get_payoffs(self):
        """Return the payoffs of the game
        1 if game won
        -(score / 50) otherwise

        Returns:
            (list): Each entry corresponds to the payoff of one player
        """
        self.payoffs = []
        if self.round.winner == -1:
            self.payoffs = [self._early_end_reward for _ in range(self.num_players)]
        elif self._use_scaled_negative_reward:
            for score in self.round.scores:
                if score == 0:
                    payoff = 1
                else:
                    payoff = self._max_negative_reward * (
                        min(self._negative_score_cutoff, score)
                        / self._negative_score_cutoff
                    )

                self.payoffs.append(payoff)
        else:
            for score in self.round.scores:
                if score == 0:
                    payoff = 1
                else:
                    payoff = self._max_negative_reward

                self.payoffs.append(payoff)

        if self._use_scaled_positive_reward and 1 in self.payoffs:
            winner = self.payoffs.index(1)
            self.payoffs[winner] = abs(min(self.payoffs))

        return self.payoffs

    def get_legal_actions(self):
        """Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        """
        return self._get_legal_actions(self.round.current_player)

    def _get_legal_actions(self, player_id):
        legal_actions = self.round.get_legal_actions(self.players, player_id)
        if not self._single_step_actions:
            return legal_actions

        joint_legal_actions = []
        if utils.YANIV_ACTION in legal_actions:
            joint_legal_actions.append(utils.YANIV_ACTION)
            legal_actions.remove(utils.YANIV_ACTION)

        for d, p in product(legal_actions, utils.pickup_actions):
            joint_legal_actions.append((d, p))

        return joint_legal_actions

    def get_player_num(self):
        """Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        """
        return self.num_players

    def get_action_num(self):
        """Return the number of applicable actions

        Returns:
            (int): The number of actions.
        """
        if self._single_step_actions:
            return len(utils.JOINED_ACTION_LIST)
        else:
            return len(utils.ACTION_LIST)

    def get_player_id(self):
        """Return the current player's id

        Returns:
            (int): current player's id
        """
        return self.round.current_player

    def is_over(self):
        """Check if the game is over

        Returns:
            (boolean): True if the game is over
        """
        return self.round.is_over

    def render(self):
        header = lambda x: "{:=^20}".format(" {} ".format(x))
        print(header("STEP {}".format(len(self.actions))))
        print("Discard Pile: {}".format(self.round.discard_pile))

        player_f = "PLAYER {}"
        cur_f = "*{}*".format(player_f)

        for player in self.players:

            print(
                header(
                    (
                        player_f
                        if player.player_id != self.round.current_player
                        else cur_f
                    ).format(player.player_id)
                )
            )

            print("Actions: {}".format(", ".join(player.actions)))
            print("Hand:    " + "  ".join([c.__str__() for c in player.hand]))

            print()


## For test
if __name__ == "__main__":
    # import time
    # random.seed(0)
    # start = time.time()
    game = YanivGame()
    for _ in range(1):
        state, button = game.init_game()
        print(button, state)
        i = 0
        while not game.is_over():
            i += 1
            legal_actions = game.get_legal_actions()
            print("legal_actions", legal_actions)
            action = np.random.choice(legal_actions)
            print("action", action)
            print()
            state, button = game.step(action)
            print(button, state)
        print(game.get_payoffs())
    print("step", i)
