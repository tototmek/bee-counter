import numpy as np

class FsmConfig:
    filter_window: int = 31
    detrend_window: int = 1269
    timeout_samples: int = 143
    static_input_threshold: float = 0.8711857628182433
    adaptive_input_q: float = 0.07
    adaptive_input_mult: float = 1.6
    adaptive_input_window: int = 5000
    use_adaptive_threshold: bool = True

    def __repr__(self):
        ret = "FSM Config:\n"
        ret += f"filter_window={self.filter_window}\n"
        ret += f"detrend_window={self.detrend_window}\n"
        ret += f"timeout_samples={self.timeout_samples}\n"
        if self.use_adaptive_threshold:
            ret += "Adaptive threshold\n"
            ret += f"    input_q={self.adaptive_input_q}\n"
            ret += f"    input_mult={self.adaptive_input_mult}\n"
            ret += f"    input_window={self.adaptive_input_window}"
        else:
            ret += "Static threshold\n"
            ret += f"    threshold value={self.static_input_threshold}"
        return ret

class FsmInput:
    def __init__(self, time: np.ndarray, signal: np.ndarray):
        self.time = time
        self.signal = signal


class FsmOutput:
    def __init__(self, enter_ts: np.ndarray, leave_ts: np.ndarray):
        self.enter_ts = enter_ts
        self.leave_ts = leave_ts

class FsmDebug:
    def __init__(self, time: np.ndarray, raw: np.ndarray, filtered: np.ndarray,
                 median: np.ndarray, detrended: np.ndarray, up_threshold: np.ndarray, bottom_threshold: np.ndarray,
                 signal_thresholded: np.ndarray, fsm_output: float):
        self.time = time
        self.raw = raw
        self.filtered = filtered
        self.median = median
        self.detrended = detrended
        self.up_threshold = up_threshold
        self.bottom_threshold = bottom_threshold
        self.signal_thresholded = signal_thresholded
        self.fsm_output = fsm_output


def increments(a: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.diff(a)))


# Fast filtering implementations (SciPy preferred, with fallbacks)
try:
    from scipy.ndimage import uniform_filter1d as _uniform_filter1d, median_filter as _median_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    _uniform_filter1d = None
    _median_filter = None


def _ensure_int(n: int) -> int:
    try:
        n = int(n)
    except Exception:
        n = 1
    return max(1, n)


def moving_median(a, n):
    a = np.asarray(a, dtype=float)
    n = _ensure_int(n)
    # Prefer SciPy
    if _HAS_SCIPY:
        # Ensure odd window for median stability
        if n % 2 == 0:
            n += 1
        return _median_filter(a, size=n, mode='nearest')
    # Fallback: use pandas rolling median (fast, C-backed)
    try:
        import pandas as _pd
        return _pd.Series(a).rolling(window=n, center=True, min_periods=1).median().to_numpy()
    except Exception:
        # Last resort: simple numpy median over shrinking window (slower)
        length = len(a)
        half = n // 2
        out = np.empty(length, dtype=float)
        for i in range(length):
            left = max(0, i - half)
            right = min(length - 1, i + half)
            out[i] = np.median(a[left:right + 1])
        return out


def moving_average(a, n):
    a = np.asarray(a, dtype=float)
    n = _ensure_int(n)
    if _HAS_SCIPY:
        return _uniform_filter1d(a, size=n, mode='nearest')
    # Fallback: fast NumPy convolution with boxcar kernel (same length)
    kernel = np.ones(n, dtype=float) / float(n)
    return np.convolve(a, kernel, mode='same')


class DetectorFsm:
    def __init__(self, timeout_samples: int):
        self.state = 0
        self.output = 0
        self.timer = 0
        self.timeout = timeout_samples

    def step(self, input: float) -> float:
        if self.state == 0:
            if input == 1:
                self.state = 1
                self.timer = 0
            elif input == -1:
                self.state = 2
                self.timer = 0

        elif self.state == 1:
            self.timer += 1
            if input == -1:
                self.state = 0
                self.output += 1
            elif self.timer > self.timeout:
                self.state = 0

        elif self.state == 2:
            self.timer += 1
            if input == 1:
                self.state = 0
                self.output -= 1
            elif self.timer > self.timeout:
                self.state = 0

        return self.output


def static_threshold(signal: np.ndarray, threshold: float):
    up_threshold = np.ones(len(signal), dtype=float) * threshold
    bottom_threshold = np.ones(len(signal), dtype=float) * -threshold
    return (up_threshold, bottom_threshold)

def adaptive_threshold(signal: np.ndarray, q: float, mult: float, window_size: int) -> np.ndarray:
    up_threshold = np.zeros(len(signal), dtype=float)
    bottom_threshold = np.zeros(len(signal), dtype=float)
    for i in range(len(signal)):
        window = signal[max(0, i - window_size // 2):min(len(signal), i + window_size // 2 + 1)]
        q_low = np.quantile(window, q)
        q_high = np.quantile(window, 1.0 - q)
        q_low_scaled = q_low * mult
        q_high_scaled = q_high * mult
        up_threshold[i] = q_high_scaled
        bottom_threshold[i] = q_low_scaled
    return (up_threshold, bottom_threshold)

def run_fsm(input: FsmInput, config: FsmConfig):
    # Filter
    signal_filtered = moving_average(input.signal, config.filter_window)

    # Detrend
    median = moving_median(signal_filtered, config.detrend_window)
    signal_detrended = signal_filtered - median

    # Threshold
    if config.use_adaptive_threshold:
        up_threshold, bottom_threshold = adaptive_threshold(signal_detrended, config.adaptive_input_q, config.adaptive_input_mult, config.adaptive_input_window)
    else:
        up_threshold, bottom_threshold = static_threshold(signal_detrended, config.static_input_threshold)
    signal_thresholded = np.zeros(len(signal_detrended), dtype=float)
    signal_thresholded[signal_detrended > up_threshold] = 1
    signal_thresholded[signal_detrended < bottom_threshold] = -1

    # Detect
    fsm = DetectorFsm(config.timeout_samples)
    fsm_output = np.zeros(len(signal_thresholded))
    enter_ts = []
    leave_ts = []
    prev_output = 0
    for i in range(len(signal_thresholded)):
        fsm_output[i] = fsm.step(signal_thresholded[i])
        if fsm_output[i] > prev_output:
            enter_ts.append(input.time[i])
        elif fsm_output[i] < prev_output:
            leave_ts.append(input.time[i])
        prev_output = fsm_output[i]

    debug = FsmDebug(
        time=input.time,
        raw=input.signal,
        filtered=signal_filtered,
        median=median,
        detrended=signal_detrended,
        up_threshold=up_threshold,
        bottom_threshold=bottom_threshold,
        signal_thresholded=signal_thresholded,
        fsm_output=fsm_output,
    )

    return FsmOutput(enter_ts, leave_ts), debug
