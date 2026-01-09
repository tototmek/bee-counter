import numpy as np
from algorithm import FsmInput, FsmOutput, increments, moving_average, moving_median, static_threshold, adaptive_threshold

class FsmConfig:
    filter_window: int = 31
    detrend_window: int = 1269
    timeout_samples: int = 143
    static_input_threshold: float = 0.8711857628182433
    adaptive_input_q: float = 0.2
    adaptive_input_mult: float = 7
    adaptive_input_amount: float = 0.4
    adaptive_input_window: int = 1500
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


def run_fsm(input: FsmInput, config: FsmConfig):
    # Filter
    signal_filtered = moving_average(input.signal, config.filter_window)

    # Detrend
    median = moving_median(signal_filtered, config.detrend_window)
    signal_detrended = signal_filtered - median

    # Threshold
    if config.use_adaptive_threshold:
        up_threshold, bottom_threshold = adaptive_threshold(signal_detrended, config.adaptive_input_q, config.adaptive_input_mult, config.adaptive_input_window, config.adaptive_input_amount, config.static_input_threshold)
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
