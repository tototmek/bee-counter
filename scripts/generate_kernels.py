import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


kernel_sizes = np.array([0.2, 0.4, 0.7, 1.4, 1.8])
n_kernels = len(kernel_sizes)


def load_kernel(path):
    with open(path, "r") as file:
        kernel = file.read().strip().split(",")
    kernel = [float(value) for value in kernel]
    return kernel


def save_kernel(path, kernel):
    with open(path, "w") as file:
        line = ",".join([str(value) for value in kernel])
        file.write(line)


original_kernel = load_kernel("data/detection_kernel.txt")
length = len(original_kernel)
kernel_sizes = length * kernel_sizes
print(kernel_sizes)


x_original = np.arange(length)

plt.figure(figsize=(12, 8))
plt.plot(original_kernel, label="Original Kernel", marker="o")

for i, new_length in enumerate(kernel_sizes):
    x_new = np.linspace(0, length - 1, int(new_length))
    f = interp1d(x_original, original_kernel, kind="linear", fill_value="extrapolate")  # Use linear interpolation
    new_kernel = f(x_new)
    plt.plot(new_kernel, label=f"Scaled Kernel {i+1} (Size {new_length})", marker="x")
    save_kernel(f"data/kernels/scaled{i}.txt", new_kernel)

save_kernel(f"data/kernels/scaled{i}.txt", new_kernel)

plt.title("Scaled Kernels")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
