import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.genfromtxt("data/measurement-2025-07-08_17-52-13.csv", delimiter=",", names=True)
N = 8

time = data["time"]
left = np.array([data[f"ch{i}_left_gate_raw"] for i in range(N)])
right = np.array([data[f"ch{i}_right_gate_raw"] for i in range(N)])
delta = left - right
time = time - time[0]
time = time / 1000

filter_window = 600


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
left_filtered = np.array([moving_average(left[i], filter_window) for i in range(N)])
right_filtered = np.array([moving_average(right[i], filter_window) for i in range(N)])
delta_filtered = left_filtered - right_filtered

# delta_filtered_hp = high_pass_filter(delta_filtered)

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

plt.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Ubuntu"],
    }
)

fig, subplots = plt.subplots(N, 1, sharex=True, figsize=(8, 12))  # Create two subplots with shared x-axis

for i, subplot in enumerate(subplots):
    subplot.plot(time_filtered, delta_filtered[i])
    subplot.set_title(f"Tunel {i}")
    # subplot.set_ylabel(r"$\mathrm{N_L}$", fontsize=14)
    subplot.grid(True)
subplots[-1].set_xlabel(r"t [s]")
fig.savefig("images/przebiegi-last.png", dpi=300)
plt.show()
