import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("data/measurement-long-processed.csv", delimiter=",", names=True)

time = data["time"]
delta = data["delta"]


def moving_median(a, n):
    ret = np.zeros(len(a) - n + 1)
    for i in range(len(a) - n + 1):
        window = a[i : i + n]
        ret[i] = np.median(window)
    return ret


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


filter_window = 2000
base_level_median = moving_median(delta, filter_window)
base_level_average = moving_average(delta, filter_window)
time_base_level = time[round(filter_window / 2) - 1 : -round(filter_window / 2)]
delta_base_level = delta[round(filter_window / 2) - 1 : -round(filter_window / 2)]

delta_base_level = delta_base_level - base_level_median

input_threshold = 1.0
fsm_input = 1.0 * (delta_base_level > input_threshold) - 1.0 * (delta_base_level < -input_threshold)

from fsm import DetectorFsm

fsm_detector = DetectorFsm()
fsm_output = np.array([fsm_detector.step(i) for i in fsm_input])
print(fsm_input)
print(fsm_output)

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 8))  # Create two subplots with shared x-axis

# First subplot (Delta)
ax1.plot(time, delta, label="Delta")
# ax1.plot(time, base_level_average, label="Base level (avg)")
ax1.plot(time_base_level, base_level_median, label="Moving median")
ax1.set_title(r"Delta raw")
ax1.set_ylabel(r"Delta")
ax1.set_xlabel(r"Time (s)")
ax1.grid(True)
ax1.legend()

ax2.plot(time_base_level, delta_base_level, label="Delta")
ax2.set_title(r"Delta de-trended")
ax2.set_ylabel(r"Delta")
ax2.set_xlabel(r"Time (s)")

ax3.plot(time_base_level, fsm_input, label="FSM input")
ax3.set_title(r"FSM input")
ax3.set_ylabel(r"FSM input")
ax3.set_xlabel(r"Time (s)")

ax4.plot(time_base_level, fsm_output, label="FSM output")
ax4.set_title(r"FSM output")
ax4.set_ylabel(r"FSM output")
ax4.set_xlabel(r"Time (s)")

plt.tight_layout()
plt.show()
