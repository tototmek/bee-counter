import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data/experiments/2025-09-20/2025-09-20_12-12-52.csv")

data["time"] = data["time"] / 1000.0
print(data)




f, ax = plt.subplots(nrows=2)
for i in range(8):
    ax[0].plot(data["time"], data[f"delta{i}"], label=f"tunel{i}")
ax[0].legend()
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("bees")
sum = np.sum(data[(f"delta{i}" for i in range(8))], axis=1)
ax[1].plot(data["time"], sum, label="suma")
ax[1].legend()
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("bees")



plt.show()