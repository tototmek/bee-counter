import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.genfromtxt("data/measurement-dead-bee.csv", delimiter=",", names=True)

time = data["time"]
delta = data["delta"]

kernel_start = 73000
kernel_end = 79000
kernel = delta[kernel_start:kernel_end]
kernel_time = time[kernel_start:kernel_end]
kernel = kernel - np.average(kernel)
kernel = -kernel
kernel = kernel[::-1]

# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#     }
# )
with open("data/detection_kernel.txt", "w") as file:
    line = ",".join([str(value) for value in kernel])
    file.write(line)


plt.figure(figsize=(10, 6))
plt.plot(kernel_time, kernel, label="Detection Kernel")

plt.title(r"$\mathrm{Detection\ Kernel}$", fontsize=16)
plt.xlabel(r"$\mathrm{Time\ (s)}$", fontsize=14)
plt.ylabel(r"$\mathrm{Gate\ Delta}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.show()
