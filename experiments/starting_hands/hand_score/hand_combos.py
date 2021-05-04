import json
import matplotlib.pyplot as plt

with open('./scores.json') as f:
    handscores = json.load(f)

score_distributions = {s: len(h) for s, h in handscores.items()}

names = list(score_distributions.keys())
values = list(score_distributions.values())

# plt.scatter(names, values)
total = sum(values)
normalized = list(map(lambda x: x / total, values))
assert sum(normalized) == 1
plt.scatter(names, normalized)

plt.show()