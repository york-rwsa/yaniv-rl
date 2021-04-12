import numpy as np
from yaniv_rl import utils

from operator import methodcaller


class YanivNoviceRuleAgent(object):
    """
    Agent always discards highest action value

    """

    def __init__(self, single_step=False):
        self.use_raw = True
        self.single_step = single_step

    def step(self, state):
        if self.single_step:
            return self._step_single(state)
        else:
            return self._step_multi(state)

    @classmethod
    def _step_single(cls, state):
        legal_actions = state["raw_legal_actions"]
        raw_state = state["raw_obs"]

        # picking up

        discard_actions = []

        if utils.YANIV_ACTION in legal_actions:
            if cls.should_yaniv(raw_state):
                discard_actions.append(utils.YANIV_ACTION)

        legal_discards = list(
            set(a[0] for a in legal_actions if a != utils.YANIV_ACTION)
        )
        discard_actions.extend(cls.best_discards(legal_discards))

        pickup_actions = cls.best_pickup_actions(raw_state["discard_pile"][-1])

        discard_action = np.random.choice(discard_actions)
        if discard_action == utils.YANIV_ACTION:
            return discard_action

        pickup_action = np.random.choice(pickup_actions)

        return (discard_action, pickup_action)

    @classmethod
    def _step_multi(cls, state):
        """Predict the action given the current state.
            Novice strategy:
                Discard stage:
                    - Yaniv if can and opponenets hand is less than my hand
                    - discard highest scoring combination of cards
                Pickup stage:
                    - draw
                    - unless ace or 2

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted
        """
        legal_actions = state["raw_legal_actions"]
        raw_state = state["raw_obs"]

        # picking up
        actions = []
        if utils.DRAW_CARD_ACTION in legal_actions:
            actions = cls.best_pickup_actions(raw_state["discard_pile"][-2])

        else:
            # discarding
            if utils.YANIV_ACTION in legal_actions:
                if cls.should_yaniv(raw_state):
                    actions.append(utils.YANIV_ACTION)

            actions.extend(
                cls.best_discards(
                    [a for a in legal_actions if a != utils.YANIV_ACTION]
                )
            )

        # if for some reason no actions are decided to be taken
        # then just pick a random legal action
        if len(actions) == 0:
            actions = legal_actions

        return np.random.choice(actions)

    @staticmethod
    def best_pickup_actions(availcards):
        """Returns the best pickup actions as follows:
        either
        [draw_card, pickup_top/bottom_card] if bottom/top card is lte 2.
        [draw_card] otherwise
        """
        actions = [utils.DRAW_CARD_ACTION]

        cardscores = list(
            map(
                methodcaller("get_score"),
                map(utils.make_card_from_str, [availcards[0], availcards[-1]]),
            )
        )
        minscore = min(cardscores)
        # if a card that scores less than 3 pick it up
        if minscore <= 2:
            idx = cardscores.index(minscore)
            if idx == 0:
                actions.append(utils.PICKUP_TOP_DISCARD_ACTION)
            elif idx == 1:
                actions.append(utils.PICKUP_BOTTOM_DISCARD_ACTION)
            else:
                raise Exception("min score not in cardscores")

        return actions

    @staticmethod
    def best_discards(legal_discard_actions):
        discard_scores = list(map(utils.score_discard_action, legal_discard_actions))
        max_discard = max(discard_scores)
        best_discards = [
            legal_discard_actions[i]
            for i, ds in enumerate(discard_scores)
            if ds == max_discard
        ]
        return best_discards

    @staticmethod
    def should_yaniv(state) -> bool:
        """decides whether or not yaniv is a good idea
        True if should yaniv
        False if should not
        """
        hand = map(utils.make_card_from_str, state["hand"])
        handscore = sum(map(methodcaller("get_score"), hand))
        # known cards is indexed from the first player then to the left
        # so since yaniv is only 2 player atm the oppoenent is always idx 1
        if len(state["known_cards"][1]) >= state["hand_lengths"][1] - 1:
            # if we know all but one of their cards
            known_cards = map(utils.make_card_from_str, state["known_cards"][1])
            known_score = sum(map(methodcaller("get_score"), known_cards))
            if handscore < known_score:
                return True
        else:
            # yaniv anyway
            return True

        return False

    def eval_step(self, state):
        """Predict the action given the current state for evaluation.
            Since the agents is not trained, this function is equivalent to step function.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (int): the action predicted by the agent
            probabilities (list): The list of action probabilities
        """
        probabilities = []
        return self.step(state), probabilities
