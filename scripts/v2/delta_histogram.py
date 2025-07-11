import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("data/measurement-processed.csv", delimiter=",", names=True)
time = data["time"]
delta = data["delta"]

delta = delta - np.average(delta)

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

plt.hist(delta, bins=200, range=(-5, 5))
plt.tight_layout()
plt.show()
