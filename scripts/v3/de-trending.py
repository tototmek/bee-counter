import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("data/measurement-2025-07-08_18-09-12.csv", delimiter=",", names=True)
N = 8

time = data["time"]
left = np.array([data[f"ch{i}_left_gate_raw"] for i in range(N)])
right = np.array([data[f"ch{i}_right_gate_raw"] for i in range(N)])
delta = left - right
time = time - time[0]
time = time / 1000


def moving_median(a, n):
    ret = np.zeros(len(a) - n + 1)
    for i in range(len(a) - n + 1):
        window = a[i : i + n]
        ret[i] = np.median(window)
    return ret


filter_window = 1000
base_level_median = np.array([moving_median(delta[i], filter_window) for i in range(N)])
time_base_level = np.array(time[round(filter_window / 2) - 1 : -round(filter_window / 2)])
delta_time_adjusted = np.array([delta[i][round(filter_window / 2) - 1 : -round(filter_window / 2)] for i in range(N)])


delta_base_level = delta_time_adjusted - base_level_median



fig, subplots = plt.subplots(N, 1, sharex=True, figsize=(8, 12))  # Create two subplots with shared x-axis

for i, subplot in enumerate(subplots):
    # subplot.plot(time_base_level, base_level_median[i])
    # subplot.plot(time, delta[i])
    subplot.plot(time_base_level, delta_base_level[i])
    subplot.set_title(f"Tunel {i}")
    # subplot.set_ylabel(r"$\mathrm{N_L}$", fontsize=14)
    subplot.grid(True)
subplots[-1].set_xlabel(r"t [s]")

plt.tight_layout()
plt.savefig("images/de-trending-8-daflkv2.png")
plt.show()
