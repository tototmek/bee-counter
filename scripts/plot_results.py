import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(
    "data/parallel-measurement-no-filter.csv", delimiter=",", names=True
)

time = data["time"]
left = data["left_gate_raw"]
right = data["right_gate_raw"]
delta = left - right

filter_window = 30


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


time_filtered = time[filter_window - 1 :]
left_filtered = moving_average(left, filter_window)
right_filtered = moving_average(right, filter_window)
delta_filtered = left_filtered - right_filtered

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

plt.figure(figsize=(10, 6))
plt.plot(time, left, label="Raw Left Gate Data")
plt.plot(time_filtered, left_filtered, label="Filtered Left Gate Data")

plt.title(r"$\mathrm{Left\ Gate\ Data}$", fontsize=16)
plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
plt.ylabel(r"$\mathrm{Gate\ Value}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, right, label="Raw Right Gate Data")
plt.plot(time_filtered, right_filtered, label="Filtered Right Gate Data")

plt.title(r"$\mathrm{Right\ Gate\ Data}$", fontsize=16)
plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
plt.ylabel(r"$\mathrm{Gate\ Value}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time, delta, label="Raw Delta")
plt.plot(time_filtered, delta_filtered, label="Filtered Delta")

plt.title(r"$\mathrm{Gate\ Delta}$", fontsize=16)
plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
plt.ylabel(r"$\mathrm{Gate\ Delta}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
