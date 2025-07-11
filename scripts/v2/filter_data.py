import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.genfromtxt("data/measurement-assembly-8ch.csv", delimiter=",", names=True)

N = 8

time = data["time"]
left = data["ch0_left_gate_raw"]
right = data["ch0_right_gate_raw"]
delta = left - right
time = time - time[0]
time = time / 1000

filter_window = 600


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


time_filtered = time[filter_window - 1 :]
left_filtered = moving_average(left, filter_window)
right_filtered = moving_average(right, filter_window)
delta_filtered = left_filtered - right_filtered

with open("data/measurement-dead-bee.csv", "w") as file:
    file.write("time,left_gate,right_gate,delta\n")
    for time, left, right, delta in zip(time_filtered, left_filtered, right_filtered, delta_filtered):
        file.write(f"{time},{left},{right},{delta}\n")
