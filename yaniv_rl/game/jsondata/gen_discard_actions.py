from itertools import combinations
import json

suits = ["C", "D", "H", "S"]
ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
deck = []
for suit in suits:
    for rank in ranks:
        deck.append(suit + rank)

# first do singles
singles = deck

print("Single card discards")
print(f"len: {len(singles)}")
print(singles)


pairs = []
# pairs
for rank in ranks:
    for (s1, s2) in combinations(suits, 2):
        p = [s1 + rank, s2 + rank]

        pairs.append("".join(sorted(p)))

print("#################")
print("pair card discards")
print(f"len: {len(pairs)}")
print(pairs)


# for triples order only matters for the middle discard
# so for every tripple there are 3 ways of putting it down
# ie for h1, d1, s1 it can be put down
# h1d1s1, d1s1h1, d1h1s1
# the two outer elements should be sorted
triples = []
for rank in ranks:
    for combo in combinations(suits, 3):
        for s in combo:
            seq = sorted([c for c in combo if c != s])
            seq.insert(1, s)
            triples.append("".join([c + rank for c in seq]))


print("#################")
print("tripple card discards")
print(f"len: {len(triples)}")
print(triples)

# similar idea for quads but the middle two cards change
quads = []
for rank in ranks:
    for combo in combinations(suits, 2):
        outer = sorted(combo)
        inner = sorted([s for s in suits if s not in outer])
        outer[1:1] = inner
        quads.append("".join([s + rank for s in outer]))

print("#################")
print("tripple card discards")
print(f"len: {len(quads)}")
print(quads)

ofakind = []
ofakind.extend(singles)
ofakind.extend(pairs)
ofakind.extend(triples)
ofakind.extend(quads)

print("**************")
print(f"total action space for * of a kind length: {len(ofakind)}")
print("**************\n\n")

# ###########
#  straights
# ###########

threecard = []
for s in suits:
    for i in range(0, 11):
        straight = [s + ranks[i], s + ranks[i + 1], s + ranks[i + 2]]

        threecard.append("".join(straight))

        # halfs part of the actionspace and not super important
        # threecard.append("".join(straight[::-1]))

print("three card straights")
print(f"len: {len(threecard)}")
print(threecard)


fourcard = []
for s in suits:
    for i in range(0, 10):
        straight = [s + ranks[i], s + ranks[i + 1], s + ranks[i + 2], s + ranks[i + 3]]

        fourcard.append("".join(straight))

        # halfs part of the actionspace and not super important
        # fourcard.append("".join(straight[::-1]))

print("#################")
print("four card straights")
print(f"len: {len(fourcard)}")
print(fourcard)

fivecard = []
for s in suits:
    for i in range(0, 9):
        straight = [
            s + ranks[i],
            s + ranks[i + 1],
            s + ranks[i + 2],
            s + ranks[i + 3],
            s + ranks[i + 4],
        ]

        fivecard.append("".join(straight))

        # halfs part of the actionspace and not super important
        # fivecard.append("".join(straight[::-1]))

print("#################")
print("five card straights")
print(f"len: {len(fivecard)}")
print(fivecard)

straights = []
straights.extend(threecard)
straights.extend(fourcard)
straights.extend(fivecard)


print("**************")
print(f"total action space for * of a kind length: {len(straights)}")
print("**************\n\n")

actionspace = []
actionspace.extend(ofakind)
actionspace.extend(straights)

print("**************")
print(f"total discard action space length: {len(actionspace)}")
print("**************")

# output actions as json dict
output = {action: i for i, action in enumerate(actionspace)}
with open("discard_actions.json", "w") as f:
    json.dump(output, f)