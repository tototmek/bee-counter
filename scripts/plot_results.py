import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.genfromtxt("data/gumis.csv", delimiter=",", names=True)

time = data["time"]
left = data["left_gate_raw"]
right = data["right_gate_raw"]
delta = left - right
time = time - time[0]
time = time / 1000

filter_window = 100


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def moving_median(a, n):
    ret = np.zeros(len(a) - n + 1)
    for i in range(len(a) - n + 1):
        window = a[i : i + n]
        ret[i] = np.median(window)
    return ret


def high_pass_filter(data, cutoff_frequency=0.05, sampling_rate=100, order=5):
    nyquist_frequency = 0.5 * sampling_rate
    normalized_cutoff = cutoff_frequency / nyquist_frequency
    b, a = signal.butter(order, normalized_cutoff, btype="high", analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def custom_filter(samples, initial_filtered_value=0, initial_sample=0):
    filtered_values = np.zeros_like(samples, dtype=float)
    filtered_values[0] = initial_filtered_value
    last_filtered_value = initial_filtered_value
    last_sample = initial_sample

    for i, sample in enumerate(samples):
        if i == 0:
            filtered_values[0] = 0.996 * (initial_filtered_value + sample - initial_sample)
            last_filtered_value = filtered_values[0]
            last_sample = sample
            continue

        filtered_value = 0.996 * (last_filtered_value + sample - last_sample)
        filtered_values[i] = filtered_value
        last_filtered_value = filtered_value
        last_sample = sample

    return filtered_values


time_filtered = time[filter_window - 1 :]
left_filtered = moving_average(left, filter_window)
right_filtered = moving_average(right, filter_window)
delta_filtered = left_filtered - right_filtered
# delta_filtered_hp = high_pass_filter(delta_filtered)

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

# plt.figure(figsize=(10, 6))
# plt.plot(time, left, label="Raw Left Gate Data")
# plt.plot(time_filtered, left_filtered, label="Filtered Left Gate Data")

# plt.title(r"$\mathrm{Left\ Gate\ Data}$", fontsize=16)
# plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
# plt.ylabel(r"$\mathrm{Gate\ Value}$", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()

# plt.figure(figsize=(10, 6))
# plt.plot(time, right, label="Raw Right Gate Data")
# plt.plot(time_filtered, right_filtered, label="Filtered Right Gate Data")

# plt.title(r"$\mathrm{Right\ Gate\ Data}$", fontsize=16)
# plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
# plt.ylabel(r"$\mathrm{Gate\ Value}$", fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.tight_layout()

plt.figure(figsize=(10, 6))
plt.plot(time, delta, label="Raw Delta")
plt.plot(time_filtered, delta_filtered, label="Filtered Delta")
# plt.plot(time_filtered, delta_filtered_hp, label="Filtered Delta High Passed")

plt.title(r"$\mathrm{Gate\ Delta}$", fontsize=16)
plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
plt.ylabel(r"$\mathrm{Gate\ Delta}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.show()
