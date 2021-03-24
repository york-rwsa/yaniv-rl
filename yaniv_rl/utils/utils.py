import os
import json
from rlcard.envs.vec_env import VecEnv
import numpy as np
from collections import OrderedDict
from yaniv_rl.game.card import YanivCard
import yaniv_rl
from typing import List
from operator import methodcaller
from tqdm import tqdm

ASSAF_PENALTY = 30
INITIAL_NUMBER_OF_CARDS = 5

PICKUP_TOP_DISCARD_ACTION = "pickup_top_discard"
PICKUP_BOTTOM_DISCARD_ACTION = "pickup_bottom_discard"
DRAW_CARD_ACTION = "draw_card"
YANIV_ACTION = "yaniv"

ROOT_PATH = yaniv_rl.__path__._path[0]

pickup_actions = [
    PICKUP_TOP_DISCARD_ACTION,
    PICKUP_BOTTOM_DISCARD_ACTION,
    DRAW_CARD_ACTION,
]
# a map of abstract action to its index and a list of abstract action
with open(
    os.path.join(ROOT_PATH, "game/jsondata/discard_actions.json"), "r"
) as file:
    DISCARD_ACTIONS = json.load(file, object_pairs_hook=OrderedDict)
    DISCARD_ACTION_LIST = list(DISCARD_ACTIONS.keys())

    ACTION_SPACE = OrderedDict()
    ACTION_SPACE.update(DISCARD_ACTIONS)

    for action in pickup_actions:
        ACTION_SPACE.update({action: len(ACTION_SPACE)})

    ACTION_SPACE.update({YANIV_ACTION: len(ACTION_SPACE)})
    ACTION_LIST = list(ACTION_SPACE.keys())


def get_hand_score(cards: List[YanivCard]) -> int:
    """Judge the score of a given cards set

    Args:
        cards (list): a list of cards

    Returns:
        score (int): the score of the given cards set
    """

    score = 0
    for card in cards:
        score += card.get_score()

    return score


def cardlist_to_action(cards: List[YanivCard]) -> str:
    return "".join([str(c) for c in cards])


def cards_to_list(cards: List[YanivCard]) -> List[str]:
    return [str(c) for c in sorted(cards, key=lambda x: (x.suit, x.get_rank_as_int()))]


def cards_to_str(cards: List[YanivCard]) -> str:
    return "".join(cards_to_list(cards))


def init_deck() -> List[YanivCard]:
    return [
        YanivCard(suit, rank) for suit in YanivCard.suits for rank in YanivCard.ranks
    ]


def make_card_from_card_id(card_id: int) -> YanivCard:
    """Make card from its card_id

    Args:
        card_id: int in range(0, 52)
    """
    if not (0 <= card_id < 52):
        raise Exception("card_id is {}: should be 0 <= card_id < 52.".format(card_id))
    rank_id = card_id % 13
    suit_id = card_id // 13
    rank = YanivCard.ranks[rank_id]
    suit = YanivCard.suits[suit_id]
    return YanivCard(rank=rank, suit=suit)


def make_card_from_str(card: str) -> YanivCard:
    """Make a card from its string repr

    Args:
        card: str like "HT" for 10 hearts or "D2" for 2 of diamonds
    """
    if len(card) != 2:
        raise ValueError("Card `{}` should be of lenght 2".format(card))
    suit = card[0]
    rank = card[1]

    return YanivCard(rank=rank, suit=suit)


def decode_cards(env_cards: np.ndarray) -> List[YanivCard]:
    result = []  # type: List[YanivCard]
    if len(env_cards) != 52:
        raise Exception("len(env_cards) is {}: should be 52.".format(len(env_cards)))
    for i in range(52):
        if env_cards[i] == 1:
            card = make_card_from_card_id(i)
            result.append(card)
    return result


def encode_cards(cards: List[YanivCard]) -> np.ndarray:
    plane = np.zeros(52, dtype=int)
    for card in cards:
        card_id = card.get_card_id()
        plane[card_id] = 1
    return plane


def score_discard_action(action: str) -> int:
    if action not in DISCARD_ACTIONS:
        raise Exception(f"action {action} not in discard list ....")

    cards_actions = [action[i : i + 2] for i in range(0, len(action), 2)]
    cards = [YanivCard(ca[0], ca[1]) for ca in cards_actions]

    return sum(map(methodcaller("get_score"), cards))


def tournament(env, num):
    """Evaluate he performance of the agents in the environment
    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.
    Returns:
        A list of average payoffs for each player
    """

    # vec env class
    vec_env = isinstance(env, VecEnv)

    payoffs = np.zeros(env.player_num, dtype=float)
    wins = np.zeros(env.player_num, dtype=int)
    assafs = np.zeros(env.player_num, dtype=int)
    draws = 0
    counter = 0
    roundlen = 0

    with tqdm(total=num, desc="Tournament: ") as bar:
        while counter < num:
            _, _payoffs = env.run(is_training=False)
            if vec_env:
                for _p in _payoffs:
                    for i, _ in enumerate(payoffs):
                        payoffs[i] += _p[i]

                    if max(_p) == 1:
                        wins[np.argmax(_p)] += 1
                    else:
                        draws += 1
                    counter += 1
            else:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _payoffs[i]

                if max(_payoffs) == 1:
                    wins[np.argmax(_payoffs)] += 1
                else:
                    draws += 1

                roundlen += len(env.game.actions)
                if env.game.round.assaf is not None:
                    assafs[env.game.round.assaf] += 1

                counter += 1

            if counter > num:
                bar.update(num - bar.n)
            else:
                bar.update(counter - bar.n)

    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter

    roundlen /= counter
    return {
        'payoffs': payoffs,
        'wins': wins,
        'draws': draws,
        'roundlen': roundlen,
        'assafs': assafs    
    }


import inspect
import contextlib


@contextlib.contextmanager
def redirect_to_tqdm():
    # Store builtin print
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print