import numpy as np


class FsmInput:
    def __init__(self, time: np.ndarray, signal: np.ndarray):
        self.time = time
        self.signal = signal


class FsmOutput:
    def __init__(self, enter_ts: np.ndarray, leave_ts: np.ndarray):
        self.enter_ts = enter_ts
        self.leave_ts = leave_ts


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

def static_threshold(signal: np.ndarray, threshold: float):
    up_threshold = np.ones(len(signal), dtype=float) * threshold
    bottom_threshold = np.ones(len(signal), dtype=float) * -threshold
    return (up_threshold, bottom_threshold)

def adaptive_threshold(signal: np.ndarray, q: float, mult: float, window_size: int, amount: float = 1, static: float = 1) -> np.ndarray:
    up_threshold = np.zeros(len(signal), dtype=float)
    bottom_threshold = np.zeros(len(signal), dtype=float)
    for i in range(len(signal)):
        window = signal[max(0, i - window_size):min(len(signal), i+1)]
        q_low = np.quantile(window, q)
        q_high = np.quantile(window, 1.0 - q)
        q_low_scaled = q_low * mult
        q_high_scaled = q_high * mult
        up_threshold[i] = q_high_scaled * amount + (1-amount)*static
        bottom_threshold[i] = q_low_scaled * amount - (1-amount)*static
    return (up_threshold, bottom_threshold)