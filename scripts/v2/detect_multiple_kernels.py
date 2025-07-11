import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("data/measurement-dead-bee.csv", delimiter=",", names=True)
kernel_paths = ["data/detection_kernel.txt"] + [f"data/kernels/scaled{i}.txt" for i in range(6)]
# kernel_paths = ["data/detection_kernel.txt"]

time = data["time"]
delta = data["delta"]


def load_kernel(path):
    with open(path, "r") as file:
        kernel = file.read().strip().split(",")
    kernel = [float(value) for value in kernel]
    return kernel


kernels = [load_kernel(path) for path in kernel_paths]


# def moving_median(a, n):
#     ret = np.zeros(len(a) - n + 1)
#     for i in range(len(a) - n + 1):
#         window = a[i : i + n]
#         ret[i] = np.median(window)
#     return ret


# filter_window = 2000
# base_level_median = moving_median(delta, filter_window)
# time = time[round(filter_window / 2) - 1 : -round(filter_window / 2)]
# delta = delta[round(filter_window / 2) - 1 : -round(filter_window / 2)]

# delta = delta - base_level_median
delta = delta - np.average(delta)

convolutions = []
for kernel in kernels:
    convolution = np.convolve(delta, kernel, "same")
    convolutions.append(convolution)

convolution = np.average(convolutions, axis=0) / len(kernel)

threshold = 0.5 * np.max(
    convolution
)  # This will need to be changed when the absolute values of convolution are known. This is gonna detect a lot whenever there are no gate passes

detection = 1.0 * (convolution > threshold) - 1.0 * (convolution < -threshold)


# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)  # Create two subplots with shared x-axis

# First subplot (Delta)
ax1.plot(time, delta, label="Delta")
ax1.set_title(r"Delta")
ax1.set_xlabel(r"Time (s)")
ax1.set_ylabel(r"Delta")
ax1.grid(True)

# Second subplot (Convolution)
ax2.plot(time, convolution, label="Convolution (average)")
# ax2.plot(time, convolution2, label="Convolution (median)")
ax2.set_title(r"Convolution")
ax2.set_xlabel(r"Time (s)")
ax2.set_ylabel(r"Convolution")
ax2.grid(True)

ax3.plot(time, detection, label="Detection")
ax3.set_title(r"Detection")
ax3.set_xlabel(r"Time (s)")
ax3.set_ylabel(r"Detection")

plt.tight_layout()
plt.savefig("../../images/detect_multiple_kernels.png")
plt.show()
