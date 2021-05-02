from itertools import combinations, permutations
import json

suits = ["D", "C", "H", "S"]
ranks = ["A","2","3","4","5","6","7","8","9","T","J","Q","K"]
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
    for (s1, s2) in permutations(suits, 2):
        pairs.append(s1+rank+s2+rank)

print("#################")
print("pair card discards")
print(f"len: {len(pairs)}")
print(pairs)

triples = []
for rank in ranks:
    for (s1, s2, s3) in permutations(suits, 3):
        triples.append(s1+rank+s2+rank+s3+rank)

print("#################")
print("tripple card discards")
print(f"len: {len(triples)}")
print(triples)

# similar idea for quads but the middle two cards change
quads = []
for rank in ranks:
    for (s1, s2, s3, s4) in permutations(suits, 4):
        quads.append(s1+rank+s2+rank+s3+rank+s4+rank)

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

# # ###########
# #  straights
# # ###########

threecard = []
for s in suits:
	for i in range(0, 11):
		straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2]]

		threecard.append("".join(straight))
		# halfs part of the actionspace and not super important
		threecard.append("".join(straight[::-1]))

print("three card straights")
print(f"len: {len(threecard)}")
print(threecard)


fourcard = []
for s in suits:
	for i in range(0, 10):
		straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2], s+ranks[i+3]]

		fourcard.append("".join(straight))
		# halfs part of the actionspace and not super important
		fourcard.append("".join(straight[::-1]))

print("#################")
print("four card straights")
print(f"len: {len(fourcard)}")
print(fourcard)

fivecard = []
for s in suits:
	for i in range(0, 9):
		straight = [s+ranks[i], s+ranks[i+1], s+ranks[i+2], s+ranks[i+3], s+ranks[i+4]]

		fivecard.append("".join(straight))
		# halfs part of the actionspace and not super important
		fivecard.append("".join(straight[::-1]))

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
