from itertools import combinations
import json
from yaniv_rl import utils
from yaniv_rl.game.card import YanivCard

suits = YanivCard.suits
ranks = YanivCard.ranks

deck = utils.init_deck()
handcombos = combinations(deck, 5)

scores = {s: [] for s in range(51)}

for hand in handcombos:
    score = utils.get_hand_score(hand)
    scores[score].append(utils.cards_to_str(hand))

with open("scores.json", "w") as f:
    json.dump(scores, f, indent=4)