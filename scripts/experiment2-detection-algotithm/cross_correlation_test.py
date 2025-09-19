import numpy as np
import matplotlib.pyplot as plt


x = np.zeros(7000)
kernel = np.sin(np.linspace(0, 2*np.pi, 141))


x[1700:1700+141] = kernel
x[4000:4000+141] = -kernel



R = np.correlate(x, kernel, mode='same')
my_R = np.zeros(x.shape)
my_online_R = np.zeros(x.shape) 

    
N = len(x)
M = len(kernel)
tau_max = N - M
for tau in range(tau_max):
    sum = 0
    for t in range(M):
        if t-tau < N and t-tau > 0:
            sum += x[t] * kernel[t-tau]
    my_R[tau] = sum


buffer = np.zeros(141)    
middle = 70
end = 141
for t in range(N):
    # push into shifting buffer
    buffer = [buffer[i+1] for i in range(end-1)] + [x[t]]
    my_online_R[t] = 0
    for tau in range(end):
        my_online_R[t] += kernel[tau] * buffer[tau]
    





fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
axes[0].plot(x, label="x")
axes[1].plot(kernel, label="kernel")
axes[2].plot(R, label="R")
axes[3].plot(R, label="my_R")
axes[4].plot(R, label="my_online_R")
for ax in axes:
    ax.legend()
plt.show()