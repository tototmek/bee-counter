import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt("data/measurement-parallel-filtered.csv", delimiter=",", names=True)

time = data["time"]
delta = data["delta"]

filter_window = 30

with open("data/detection_kernel.txt", "r") as file:
    kernel = file.read().strip().split(",")
    kernel = [float(value) for value in kernel]

delta = delta - np.average(delta)

convolution = np.convolve(delta, kernel, "same")
convolution = convolution / len(kernel)

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)  # Create two subplots with shared x-axis

# First subplot (Delta)
ax1.plot(time, delta, label="Delta")
ax1.set_title(r"$\mathrm{Delta}$", fontsize=16)
ax1.set_ylabel(r"$\mathrm{Delta}$", fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True)

# Second subplot (Convolution)
ax2.plot(time, convolution, label="Convolution")
ax2.set_title(r"$\mathrm{Convolution}$", fontsize=16)
ax2.set_xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
ax2.set_ylabel(r"$\mathrm{Convolution}$", fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True)

plt.tight_layout()
plt.show()
