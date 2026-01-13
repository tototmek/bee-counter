import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

from algorithm import FsmInput, FsmOutput, increments, moving_average, moving_median, static_threshold, adaptive_threshold

class CorrelationConfig:
    filter_window: int = 20
    detrend_window: int = 850
    kernel_scales = [] # list of additional scaled kernels. Leave empty for one kernel mode
    static_input_threshold: float = 95
    adaptive_input_q: float = 0.3
    adaptive_input_mult: float = 9
    adaptive_input_amount: float = 0.4
    adaptive_input_window: int = 1500
    use_adaptive_threshold: bool = True

    def __repr__(self):
        ret = "Correlation Config:\n"
        ret += f"filter_window={self.filter_window}\n"
        ret += f"detrend_window={self.detrend_window}\n"
        ret += f"kernel_scales={self.kernel_scales}\n"
        if self.use_adaptive_threshold:
            ret += "Adaptive threshold\n"
            ret += f"    input_q={self.adaptive_input_q}\n"
            ret += f"    input_mult={self.adaptive_input_mult}\n"
            ret += f"    input_window={self.adaptive_input_window}"
        else:
            ret += "Static threshold\n"
            ret += f"    threshold value={self.static_input_threshold}"
        return ret

class CorrelationDebug:
    def __init__(self, time: np.ndarray, raw: np.ndarray, filtered: np.ndarray,
                 median: np.ndarray, detrended: np.ndarray, up_threshold: np.ndarray, bottom_threshold: np.ndarray, correlations, total_correlation, kernels):
        self.time = time
        self.raw = raw
        self.filtered = filtered
        self.median = median
        self.detrended = detrended
        self.correlations = correlations
        self.total_correlation = total_correlation
        self.kernels = kernels
        self.up_threshold = up_threshold
        self.bottom_threshold = bottom_threshold

def scale_kernel(kernel: np.ndarray, scale: float):
    original_length = len(kernel)
    new_length = int(original_length * scale)
    original_indices = np.arange(original_length)
    new_indices = np.linspace(0, original_length - 1, new_length)
    return np.interp(new_indices, original_indices, kernel)

def run_correlation(input: FsmInput, config: CorrelationConfig):
    # Filter
    signal_filtered = moving_average(input.signal, config.filter_window)

    # Detrend
    median = moving_median(signal_filtered, config.detrend_window)
    signal_detrended = signal_filtered - median

    kernel = np.load("scripts/experiment2-detection-algotithm/data/kernel-synthetic.npy")

    kernels = [kernel]
    for scale in config.kernel_scales:
        kernels.append(scale_kernel(kernel, scale))

    # Run convolution
    correlated_signals = []
    for kernel in kernels:
        correlated_signals.append(signal.correlate(signal_detrended, kernel, mode='same'))
    correlated_signal = np.mean(np.array(correlated_signals), axis=0)
    # correlated_signal = np.max(np.array(correlated_signals), axis=0)

    if config.use_adaptive_threshold:
        up_threshold, bottom_threshold = adaptive_threshold(correlated_signal, config.adaptive_input_q, config.adaptive_input_mult, config.adaptive_input_window, config.adaptive_input_amount, config.static_input_threshold)
    else:
        up_threshold, bottom_threshold = static_threshold(correlated_signal, config.static_input_threshold)


    above_threshold = correlated_signal >= up_threshold
    below_threshold = correlated_signal <= bottom_threshold

    previous_above_state = np.insert(above_threshold[:-1], 0, False)
    previous_below_state = np.insert(below_threshold[:-1], 0, False)
    above_rising_edges = above_threshold & ~previous_above_state
    below_rising_edges = below_threshold & ~previous_below_state
    enter_ts = input.time[np.where(below_rising_edges)[0]]
    leave_ts = input.time[np.where(above_rising_edges)[0]]

    debug = CorrelationDebug(
        time=input.time,
        raw=input.signal,
        filtered=signal_filtered,
        median=median,
        detrended=signal_detrended,
        correlations=correlated_signals,
        total_correlation=correlated_signal,
        kernels = kernels,
        up_threshold=up_threshold,
        bottom_threshold=bottom_threshold
    )

    return FsmOutput(enter_ts, leave_ts), debug

def setup_latex_plots():
    latex_preamble = r"""
        \usepackage[utf8]{inputenc} 
        \usepackage[T1]{fontenc}
        """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": latex_preamble,
        "font.size": 10
        # "mathtext.fontset": "cm" 
    })


if __name__ == "__main__":
    # Extract the kernel:

    import pandas as pd
    import matplotlib.pyplot as plt
    channel_idx = 0
    RAW_DATA_FILE = 'data/experiments/processed-data/raw-time-adjusted.csv'
    channel_name = f'delta{channel_idx}'
    df = pd.read_csv(RAW_DATA_FILE)
    mask = np.ones(len(df), dtype=bool)
    signal = df.loc[mask, channel_name].to_numpy(dtype=float)

    signal = moving_average(signal, 50)
    print("{"+",".join(signal[0:10000].astype(str))+"};")
    median = moving_median(signal, 1050)
    signal = signal-median

    KERNEL_START = 3103
    KERNEL_LENGTH = 600
    kernel_end = KERNEL_START + KERNEL_LENGTH
    kernel = signal[KERNEL_START:kernel_end]

    synthetic_kernel = np.zeros(121)
    sin = np.sin(np.linspace(0, 2*np.pi, 81))
    # fun = np.power(sin, 2) * np.sign(sin) * -1.33
    fun = sin * -1.33
    synthetic_kernel[20:101] = fun
    # print("{"+",".join(synthetic_kernel.astype(str))+"};")

    setup_latex_plots()

    plt.figure(figsize=(3.5, 1.5))
    plt.plot(kernel, c="grey", lw=0.75)
    plt.plot(synthetic_kernel, c="black", lw=0.75)
    plt.xlabel(r'$m$')
    plt.ylabel(r'$w(m)$')
    plt.legend(['(a)', '(b)'], frameon=False)
    plt.savefig('kernels.pdf', bbox_inches="tight")
    plt.show()
    np.save("scripts/experiment2-detection-algotithm/data/kernel.npy", kernel)
    np.save("scripts/experiment2-detection-algotithm/data/kernel-synthetic.npy", synthetic_kernel)