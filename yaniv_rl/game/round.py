from os import execv
from rlcard.core import Card

from yaniv_rl import utils
from yaniv_rl.game.player import YanivPlayer
from yaniv_rl.game.dealer import YanivDealer

from itertools import groupby, combinations
from typing import List


class YanivRound(object):
    def __init__(
        self,
        dealer: YanivDealer,
        num_players,
        np_random,
        starting_player=0,
    ):
        """Initialize the round class

        Args:
            dealer (object): the object of YanivDealer
            num_players (int): the number of players in game
        """
        self.np_random = np_random
        self.dealer = dealer
        self.current_player = starting_player
        self.num_players = num_players
        self.discard_pile = []  # List[List[Card]]

        self.known_cards = [[] for x in range(num_players)]

        # discard first
        self.discarding = True

        self.is_over = False
        self.winner = None
        self.scores = None
        self.assaf = None
        
        self.deck_replacements = 0

    def proceed_round(self, players, action):
        """Call other Classes's functions to keep one round running

        Args:
            player (object): YanivPlayer
            action (str): string of legal action
        """
        if not self.discarding:
            if action == utils.DRAW_CARD_ACTION:
                self._perform_draw_action(players)
                self._next_player()
                return None

            if action == utils.PICKUP_TOP_DISCARD_ACTION:
                self._perform_pickup_up_top_card_action(players)
                self._next_player()
                return None

            if action == utils.PICKUP_BOTTOM_DISCARD_ACTION:
                self._perform_pickup_up_bottom_card_action(players)
                self._next_player()
                return None
        else:
            if action == utils.YANIV_ACTION:
                self._perform_yaniv_action(players)
                return None

            if action in utils.DISCARD_ACTION_LIST:
                self._perform_discard_action(players, action)
                self.discarding = False
                return None

        raise Exception("invalid yaniv action, tried {}, discarding: {}".format(action, self.discarding))

    def get_legal_actions(self, players: List[YanivPlayer], player_id):
        if not self.discarding:
            return utils.pickup_actions

        hand = players[player_id].hand
        legal_actions = []

        if utils.get_hand_score(hand) <= 7:
            legal_actions.append("yaniv")

        # can discard single cards
        for card in hand:
            legal_actions.append(card.suit + card.rank)

        suitKey = lambda c: c.suit
        rankKey = lambda c: c.get_rank_as_int()
        # groups of rank
        for rank, group in groupby(sorted(hand, key=rankKey), key=rankKey):
            group = sorted(list(group), key=suitKey)

            if len(group) == 1:
                continue

            if len(group) >= 2:
                # combinations of 2 cards
                for combo in combinations(group, 2):
                    legal_actions.append(utils.cardlist_to_action(combo))

            if len(group) >= 3:
                for combo in combinations(group, 3):
                    for c in combo:
                        seq = [s for s in combo if c != s]
                        seq.insert(1, c)
                        legal_actions.append(utils.cardlist_to_action(seq))

            if len(group) == 4:
                for combo in combinations(group, 2):
                    outer = list(combo)
                    inner = [c for c in group if c not in outer]
                    outer[1:1] = inner
                    legal_actions.append(utils.cardlist_to_action(outer))

        # straights
        for suit, group in groupby(sorted(hand, key=suitKey), key=suitKey):
            cards = sorted(group, key=rankKey)

            for _, straight in groupby(
                enumerate(cards), key=lambda x: x[0] - x[1].get_rank_as_int()
            ):
                straight = list(straight)
                if len(straight) < 3:
                    continue

                straight = [s[1] for s in straight]

                # whole straight
                legal_actions.append(utils.cardlist_to_action(straight))

                if len(straight) >= 4:
                    legal_actions.append(utils.cardlist_to_action(straight[0:3]))
                    legal_actions.append(utils.cardlist_to_action(straight[1:4]))

                if len(straight) == 5:
                    legal_actions.append(utils.cardlist_to_action(straight[2:5]))

                    legal_actions.append(utils.cardlist_to_action(straight[0:4]))
                    legal_actions.append(utils.cardlist_to_action(straight[1:5]))

        return legal_actions

    def get_state(self, players, player_id):
        """Get player's state

        Args:
            players (list): The list of player
            player_id (int): The id of the player
        """
        state = {}
        player = players[player_id]

        state["hand"] = utils.cards_to_list(player.hand)

        state["discard_pile"] = [
            utils.cards_to_list(cards) for cards in self.discard_pile
        ]

        # known cards starts at the current player index
        state["known_cards"] = [
            utils.cards_to_list(cards)
            for cards in self.known_cards[player_id:] + self.known_cards[:player_id]
        ]
        # also indexed from current player
        state["hand_lengths"] = [
            len(p.hand) for p in players[player_id:] + players[:player_id]
        ]

        return state

    def replace_deck(self):
        """Add cards have been played to deck"""
        top_discard = [self.discard_pile.pop()]
        self.dealer.deck.extend((card for d in self.discard_pile for card in d))
        self.dealer.shuffle()
        self.discard_pile = top_discard
        self.deck_replacements += 1


    def flip_top_card(self):
        self.discard_pile.append([self.dealer.draw_card()])

    def get_next_player(self):
        return self._get_next_player(self.current_player)

    def _get_next_player(self, player_id):
        return (player_id + 1) % self.num_players

    def _next_player(self):
        """increments the player counter"""
        self.current_player = self.get_next_player()
        self.discarding = True

    def _perform_draw_action(self, players):
        if not self.dealer.deck:
            self.replace_deck()

        card = self.dealer.deck.pop()
        players[self.current_player].hand.append(card)

        # TODO deal with itsbah

    def _perform_pickup_up_top_card_action(self, players):
        self._pickup_card_from_discard_pile(players, top=True)

    def _perform_pickup_up_bottom_card_action(self, players):
        self._pickup_card_from_discard_pile(players, top=False)

    def _pickup_card_from_discard_pile(self, players, top):
        # discard_pile[-2] because the step before the player discards
        # otherwise they pick up their own card
        prev_discard = self.discard_pile[-2]
        if top:
            card = prev_discard.pop()
        else:
            card = prev_discard.pop(0)

        if len(prev_discard) == 0:
            self.discard_pile.remove(prev_discard)

        players[self.current_player].hand.append(card)
        self.known_cards[self.current_player].append(card)

    def _perform_yaniv_action(self, players):
        winner = None
        scores = []
        for player in players:
            scores.append(utils.get_hand_score(player.hand))

        # assaf
        if any(
            (
                score <= scores[self.current_player]
                for i, score in enumerate(scores)
                if i != self.current_player
            )
        ):
            print("player {} assaf".format(self.current_player))
            self.assaf = self.current_player
            scores[self.current_player] += utils.ASSAF_PENALTY

            # the winner is the player with the lowest score closest
            # to the right of the current player
            minScore = min(scores)
            winnerIndex = self.current_player
            while scores[winnerIndex] != minScore:
                winnerIndex -= 1
                if winnerIndex < 0:
                    winnerIndex = len(scores) - 1

            scores[winnerIndex] = 0
            winner = winnerIndex
        else:
            winner = self.current_player
            scores[self.current_player] = 0

        self.winner = winner
        self.scores = scores
        self.is_over = True

    def _perform_discard_action(self, players, action):
        discard = []
        current_hand = players[self.current_player].hand

        toDiscard = [action[i : i + 2] for i in range(0, len(action), 2)]
        discard_len = len(toDiscard)

        cardsToDiscard = []
        for card in current_hand:
            for d in toDiscard:
                if card.suit == d[0] and card.rank == d[1]:
                    cardsToDiscard.append(card)
                    toDiscard.remove(d)
                    break

            if len(toDiscard) == 0:
                break
            
        if len(cardsToDiscard) != discard_len:
            raise Exception("Cannot discard all cards. Trying action {} on hand {}".format(action, current_hand))

        for card in cardsToDiscard:
            current_hand.remove(card)
            if card in self.known_cards[self.current_player]:
                self.known_cards[self.current_player].remove(card)

        self.discard_pile.append(cardsToDiscard)
