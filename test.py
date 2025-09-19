from data import signal, x
import matplotlib.pyplot as plt



fig, axes = plt.subplots(2, 1, figsize=(14, 13), sharex=True)
axes[0].plot(x)
axes[1].plot(signal)
plt.show()