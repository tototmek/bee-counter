import serial
import time
import serial.tools.list_ports


# Acquire data from the serial port

N = 8
baudrate = 115200

# Set output path to the current date and time
output_path = f"data/measurement-{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

# Set the names of the columns
names = ["time"]
for i in range(N):
    names += [f"ch{i}_left_gate_raw", f"ch{i}_right_gate_raw"]

try:
    # Try opening all of the serial ports available:
    for serial_port in serial.tools.list_ports.comports():
        try:
            print(f"Opening {serial_port.device}...")
            ser = serial.Serial(serial_port.device, baudrate)
            print(f"Opened {serial_port.device}.")
            break
        except serial.SerialException as e:
            continue

    time.sleep(0.2)
    ser.flush()
    ser.read_all()
    with open(output_path, "w") as file:
        file.writelines(",".join(names) + "\n")
        while True:
            if ser.in_waiting > 0:
                data = ser.readline().decode("utf-8").strip()
                print(data)
                file.writelines([data + "\n"])
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
except KeyboardInterrupt:
    print("Exiting serial reader.")
finally:
    if "ser" in locals() and ser.is_open:
        ser.close()




import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data = np.genfromtxt(output_path, delimiter=",", names=True)

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

time_filtered = time[filter_window - 1 :]
left_filtered = np.array([moving_average(left[i], filter_window) for i in range(N)])
right_filtered = np.array([moving_average(right[i], filter_window) for i in range(N)])
delta_filtered = left_filtered - right_filtered


# plt.rcParams.update(
#     {
#         # "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Ubuntu"],
#     }
# )


fig, subplots = plt.subplots(N, 1, sharex=True, figsize=(8, 12))  # Create two subplots with shared x-axis

for i, subplot in enumerate(subplots):
    subplot.plot(time_filtered, delta_filtered[i])
    subplot.set_title(f"Tunel {i}")
    subplot.grid(True)
subplots[-1].set_xlabel(r"t [s]")

plt.tight_layout()
plt.show()
