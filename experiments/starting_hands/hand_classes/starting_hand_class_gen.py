from itertools import combinations
import json

suits = ["D", "C", "H", "S"]
ranks = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]

deck = []
for suit in suits:
    for rank in ranks:
        deck.append(suit + rank)

def hand_to_str(hand):
    return "".join((sorted(hand)))

hands = {}
used_hands = set()
# 5S   - Five card straight
hands['5S'] = []
for s in suits:
    for i in range(0, 9):
        straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2], s+ranks[i+3], s+ranks[i+4]]
        hands['5S'].append(hand_to_str(straight))
used_hands.update(hands['5S'])

print("==== five card straight ====")
print("len: ", len(hands['5S']))

# 4K   - Four of a kind
hands['4K'] = []
for rank in ranks:
    combo = [s + rank for s in suits]
    for card in filter(lambda c: c not in combo, deck):
        hands['4K'].append(hand_to_str(combo + [card]))
used_hands.update(hands['4K'])

print("==== Four of a kind ====")
print("len: ", len(hands['4K']))

# 4S   - Four card straight
hands['4S'] = []
for s in suits:
    for i in range(0, 10):
        straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2], s+ranks[i+3]]
        for card in filter(lambda c: c not in straight, deck):
            hand = hand_to_str(straight + [card])
            if hand not in used_hands:
                hands['4S'].append(hand)
                used_hands.add(hand)


print("==== Four card straight ====")
print("len: ", len(hands['4S']))

# 3S2K - Three card straight + pair
hands['3S2K'] = []
for s in suits:
    for i in range(0, 11):
        straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2]]
        for c1, c2 in combinations([c for c in deck if c not in straight], 2):
            # of same rank
            if c1[1] == c2[1]:
                hand = hand_to_str(straight + [c1, c2])
                if hand not in used_hands:
                    hands['3S2K'].append(hand)
                    used_hands.add(hand)

print("==== Three card straight + pair ====")
print("len: ", len(hands['3S2K']))

# 3K2K - Three of a kind + Pair
hands['3K2K'] = []
for rank in ranks:
    for suit_combo in combinations(suits, 3):
        combo = [s + rank for s in suit_combo]
        for c1, c2 in combinations([c for c in deck if c not in combo], 2):
            # of same rank
            if c1[1] == c2[1]:
                hand = hand_to_str(combo + [c1, c2])
                # if hand == 'DJSJSJSKSQ':
                #     print(rank, suit_combo, combo, c1, c2, hand)
                if hand not in used_hands:
                    hands['3K2K'].append(hand)
                    used_hands.add(hand)

print("==== Three of a kind + Pair ====")
print("len: ", len(hands['3K2K']))

# 3S   - Three card straight
hands['3S'] = []
for s in suits:
    for i in range(0, 11):
        straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2]]
        for c1, c2 in combinations(filter(lambda c: c not in straight, deck), 2):
            # not a piar
            if c1[1] != c2[1]:
                hand = hand_to_str(straight + [c1, c2])
                if hand not in used_hands:
                    hands['3S'].append(hand)
                    used_hands.add(hand)
                    

print("==== Three card straight ====")
print("len: ", len(hands['3S']))

# 3K   - Three of a kind
hands['3K'] = []
for rank in ranks:
    for suit_combo in combinations(suits, 3):
        combo = [s + rank for s in suit_combo]
        for c1, c2 in combinations(filter(lambda c: c not in combo, deck), 2):
            # not a pair
            if c1[1] != c2[1]:
                hand = hand_to_str(combo + [c1, c2])
                if hand not in used_hands:
                    hands['3K'].append(hand)
                    used_hands.add(hand)
                    

print("==== Three of a kind ====")
print("len: ", len(hands['3K']))

# 2K2K - Two pair
hands['2K2K'] = []
for rank1 in ranks:
    for (s1, s2) in combinations(suits, 2):
        pair1 = [s1 + rank1, s2 + rank1]

        for rank2 in (r for r in ranks if r != rank1):
            for (s1, s2) in combinations(suits, 2):
                pair2 = [s1 + rank2, s2 + rank2]

                for card in deck:
                    if card in pair1 or card in pair2 or \
                       card[1] == rank1 or card[1] == rank2:
                        continue
                    
                    hand = hand_to_str(pair1 + pair2 + [card])
                    if hand in used_hands:
                        hands['2K2K'].append(hand)
                        used_hands.add(hand)

print("==== Two pair ====")
print("len: ", len(hands['2K2K']))

# 2K   - Pair
hands['2K'] = []
for rank in ranks:
    for (s1, s2) in combinations(suits, 2):
        pair = [s1 + rank, s2 + rank]
        for cards in combinations(filter(lambda c: c not in pair, deck), 3):
            hand = hand_to_str(pair + list(cards))

            if hand not in used_hands:
                hands["2K"].append(hand)
                used_hands.add(hand)

print("==== pair ====")
print("len: ", len(hands['2K']))

# x    - High card
hands['X'] = []
for cards in combinations(deck, 5):
    hand = hand_to_str(cards)
    if hand not in used_hands:
        hands['X'].append(hand)
        used_hands.add(hand)
        
print("==== High card ====")
print("len: ", len(hands['X']))

print("TOTAL : ", sum(len(x) for x in hands.values()))

allitems = set()

for k, v in hands.items():
    print("len({}) = {}".format(k, len(v)))
    # print(set(v) & allitems)
    allitems.update(v)

print("len all ", len(allitems))


with open("hand_classes.json", "w") as f:
    json.dump(hands, f, indent=4)