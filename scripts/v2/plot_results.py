import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.genfromtxt("data/measurement-dead-bee-raw.csv", delimiter=",", names=True)

time = data["time"]
left = data["left_gate_raw"]
right = data["right_gate_raw"]
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

plt.rcParams.update(
    {
        # "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Ubuntu"],
    }
)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))  # Create two subplots with shared x-axis
ax1.plot(time, left, label="Sygnał oryginalny")
ax1.plot(time_filtered, left_filtered, label="Sygnał filtrowany")
ax1.set_title("Bramka lewa")
ax1.set_ylabel(r"$\mathrm{N_L}$", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True)
ax2.plot(time, right, label="Sygnał oryginalny")
ax2.plot(time_filtered, right_filtered, label="Sygnał filtrowany")
ax2.set_title("Bramka prawa")
ax2.set_xlabel(r"t [s]")
ax2.set_ylabel(r"$\mathrm{N_R}$", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True)
fig.savefig("../../images/przebiegiLR.png", dpi=300)
fig.show()

plt.figure(figsize=(8, 5))
plt.plot(time, delta, label="Sygnał oryginalny")
plt.plot(time_filtered, delta_filtered, label="Sygnał filtrowany")
plt.title("Różnica sygnałów z obu bramek")
plt.xlabel(r"t [s]")
plt.ylabel(r"$\mathrm{N_L} - \mathrm{N_R}$", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../images/przebiegiDelta.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(time_filtered[60000:150000], delta_filtered[60000:150000])
plt.title("Różnica sygnałów z obu bramek (dane filtrowane)")
plt.xlabel(r"t [s]")
plt.ylabel(r"$\mathrm{N_L} - \mathrm{N_R}$", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("../../images/przebiegiDeltaZoom.png", dpi=300)
plt.show()
