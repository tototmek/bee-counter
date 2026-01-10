#!/usr/bin/env python3

import argparse
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fsm import FsmInput, FsmConfig, run_fsm
from convolve import CorrelationConfig, run_correlation


RAW_DATA_FILE = 'data/experiments/processed-data/raw-time-adjusted.csv'
ANNOTATION_DIR = 'data/experiments/manual-annotation'


def load_signal(channel_idx: int, start_time: Optional[float], end_time: Optional[float],
                raw_data_file: str = RAW_DATA_FILE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load channel signal data and corresponding time within a given time range.

    Args:
        channel_idx: Channel index (0-7)
        start_time: Start time in seconds (inclusive) or None for full range
        end_time: End time in seconds (inclusive) or None for full range
        raw_data_file: Path to raw data CSV containing 'time' and 'delta{idx}' columns

    Returns:
        (time, signal) as numpy arrays within [start_time, end_time] or full range
    """
    if not os.path.exists(raw_data_file):
        raise FileNotFoundError(f"Raw data not found: {raw_data_file}")

    channel_name = f'delta{channel_idx}'
    df = pd.read_csv(raw_data_file)
    if 'time' not in df.columns or channel_name not in df.columns:
        raise ValueError(f"Raw data must contain 'time' and channel column '{channel_name}'")

    if start_time is None or end_time is None:
        mask = np.ones(len(df), dtype=bool)
    else:
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)

    times = df.loc[mask, 'time'].to_numpy(dtype=float)
    values = df.loc[mask, channel_name].to_numpy(dtype=float)
    return times, values


def load_annotation_timestamps(channel_idx: int, start_time: float, end_time: float,
                               annotation_dir: str = ANNOTATION_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load annotation timestamps within a given time range as two numpy arrays.

    Args:
        channel_idx: Channel index (0-7)
        start_time: Start time in seconds (inclusive)
        end_time: End time in seconds (inclusive)
        annotation_dir: Directory containing 'tunel{idx}.csv' files with 'timestamp,event_type'

    Returns:
        (enter_timestamps, leave_timestamps) as numpy float arrays
    """
    ann_file = os.path.join(annotation_dir, f'tunel{channel_idx}.csv')
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    ann = pd.read_csv(ann_file)
    if not {'timestamp', 'event_type'}.issubset(ann.columns):
        raise ValueError("Annotation CSV must contain 'timestamp' and 'event_type'")

    mask = (ann['timestamp'] >= start_time) & (ann['timestamp'] <= end_time)
    ann = ann.loc[mask]

    enters = ann[ann['event_type'] == 'enter']['timestamp'].to_numpy(dtype=float)
    leaves = ann[ann['event_type'] == 'leave']['timestamp'].to_numpy(dtype=float)
    return enters, leaves


def compute_event_metrics(gt_ts: np.ndarray, pred_ts: np.ndarray, tol_s: float) -> Tuple[int, int, int]:
    """Compute TP, FP, FN between two sorted arrays using tolerance in seconds."""
    gt = np.sort(gt_ts.astype(float))
    pred = np.sort(pred_ts.astype(float))
    i = j = 0
    tp = fp = fn = 0
    while i < len(gt) and j < len(pred):
        dt = pred[j] - gt[i]
        if abs(dt) <= tol_s:
            tp += 1
            i += 1
            j += 1
        elif pred[j] < gt[i] - tol_s:
            fp += 1
            j += 1
        else:  # gt[i] < pred[j] - tol_s
            fn += 1
            i += 1
    # Remaining unmatched
    fp += max(0, len(pred) - j)
    fn += max(0, len(gt) - i)
    return tp, fp, fn


def unmatched_events(gt_ts: np.ndarray, pred_ts: np.ndarray, tol_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return unmatched ground-truth and unmatched predictions using greedy matching within tol_s."""
    gt = np.sort(gt_ts.astype(float))
    pred = np.sort(pred_ts.astype(float))
    i = j = 0
    matched_gt = []
    matched_pred = []
    while i < len(gt) and j < len(pred):
        dt = pred[j] - gt[i]
        if abs(dt) <= tol_s:
            matched_gt.append(i)
            matched_pred.append(j)
            i += 1
            j += 1
        elif pred[j] < gt[i] - tol_s:
            j += 1
        else:
            i += 1
    unmatched_gt_idx = np.setdiff1d(np.arange(len(gt)), np.asarray(matched_gt, dtype=int), assume_unique=False)
    unmatched_pred_idx = np.setdiff1d(np.arange(len(pred)), np.asarray(matched_pred, dtype=int), assume_unique=False)
    return gt[unmatched_gt_idx], pred[unmatched_pred_idx]


def error_density_series(time: np.ndarray, error_ts: np.ndarray, margin_s: float = 10.0) -> np.ndarray:
    """Compute for each time point the number of errors within ±margin_s seconds."""
    if error_ts.size == 0 or time.size == 0:
        return np.zeros_like(time, dtype=float)
    errs = np.sort(error_ts.astype(float))
    left = 0
    right = 0
    counts = np.zeros_like(time, dtype=float)
    for k, t in enumerate(time):
        # advance left to maintain errs[left] >= t - margin
        while left < len(errs) and errs[left] < t - margin_s:
            left += 1
        # advance right to maintain errs[right] <= t + margin
        while right < len(errs) and errs[right] <= t + margin_s:
            right += 1
        counts[k] = right - left
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description='FSM runner: load channel, run FSM, compare to annotations')
    parser.add_argument('--channel', '-c', type=int, required=True, help='Channel index (0-7)')
    parser.add_argument('--start', '-s', type=float, default=None, help='Start time in seconds (default: full)')
    parser.add_argument('--end', '-e', type=float, default=None, help='End time in seconds (default: full)')
    parser.add_argument('--raw', type=str, default=RAW_DATA_FILE, help='Path to raw data CSV')
    parser.add_argument('--ann_dir', type=str, default=ANNOTATION_DIR, help='Path to annotation directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--tol', type=float, default=3.0, help='Tolerance (seconds) for event matching (default: 3.0)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.channel < 0 or args.channel > 7:
        print("Error: Channel index must be between 0 and 7")
        return 1
    if (args.start is not None and args.end is not None) and (args.start >= args.end):
        print("Error: Start time must be less than end time")
        return 1

    try:
        time, signal = load_signal(args.channel, args.start, args.end, args.raw)
        if time.size == 0:
            print("Error: No data in selected range")
            return 1
        # Effective range used
        eff_start = float(time[0])
        eff_end = float(time[-1])
        enter_ts, leave_ts = load_annotation_timestamps(args.channel, eff_start, eff_end, args.ann_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Loaded signal samples: {signal.size}")
    print(f"Time samples: {time.size}")
    print(f"Enter events in range: {enter_ts.size}")
    print(f"Leave events in range: {leave_ts.size}")
    fsm_input = FsmInput(time, signal)

    # algorithm = "fsm"
    algorithm = "correlation"
    
    if algorithm == 'fsm':
        config = FsmConfig()
        fsm_output, debug = run_fsm(fsm_input, config)
    elif algorithm == 'correlation':
        config = CorrelationConfig()
        fsm_output, debug = run_correlation(fsm_input, config)

    # Metrics
    tp_e, fp_e, fn_e = compute_event_metrics(enter_ts, np.asarray(fsm_output.enter_ts, dtype=float), args.tol)
    tp_l, fp_l, fn_l = compute_event_metrics(leave_ts, np.asarray(fsm_output.leave_ts, dtype=float), args.tol)
    precision_e = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0.0
    recall_e = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
    precision_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
    recall_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
    f1_e = 2 * precision_e * recall_e / (precision_e + recall_e) if (precision_e + recall_e) > 0 else 0.0
    f1_l = 2 * precision_l * recall_l / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0.0
    avg_f1 = (f1_e + f1_l) / 2

    print(config)

    print("\nMetrics (tolerance = %.2fs):" % args.tol)
    print("Enter: TP=%d FP=%d FN=%d\tPrecision=%.2f Recall=%.2f F1=%.2f" % (tp_e, fp_e, fn_e, precision_e, recall_e, f1_e))
    print("Leave: TP=%d FP=%d FN=%d\tPrecision=%.2f Recall=%.2f F1=%.2f" % (tp_l, fp_l, fn_l, precision_l, recall_l, f1_l))
    print(f"Average F1={avg_f1}")

    # Error density (hardcoded margin 10s)
    unmatched_enter_gt, unmatched_enter_pred = unmatched_events(enter_ts, np.asarray(fsm_output.enter_ts, dtype=float), args.tol)
    unmatched_leave_gt, unmatched_leave_pred = unmatched_events(leave_ts, np.asarray(fsm_output.leave_ts, dtype=float), args.tol)
    error_ts = np.sort(np.concatenate([unmatched_enter_gt, unmatched_enter_pred, unmatched_leave_gt, unmatched_leave_pred]))
    err_density = error_density_series(debug.time, error_ts, margin_s=30.0)

    if args.no_plot:
        return 0

    # Plot debug outputs
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    fig.suptitle('FSM Preprocessing', fontsize=14)

    ax = axes[0]
    ax.plot(debug.time, debug.raw, color='#1f77b4', alpha=0.35, linewidth=0.8, label='Raw')
    ax.plot(debug.time, debug.filtered, color='#1f77b4', alpha=0.95, linewidth=1.6, label='Filtered (MA)')
    # Overlay original annotations as pins on the same axis (axes fraction for Y)
    xform = ax.get_xaxis_transform()
    if enter_ts.size > 0:
        ax.vlines(enter_ts, 0, 1, transform=xform, colors='green', linewidth=1.5, alpha=0.8, label='Ann Enter')
        ax.scatter(enter_ts, np.full_like(enter_ts, 0.95, dtype=float), transform=xform, color='green', s=35, zorder=5)
    if leave_ts.size > 0:
        ax.vlines(leave_ts, 0, 1, transform=xform, colors='red', linewidth=1.5, alpha=0.8, label='Ann Leave')
        ax.scatter(leave_ts, np.full_like(leave_ts, 0.05, dtype=float), transform=xform, color='red', s=35, zorder=5)
    ax.set_ylabel('Signal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    ax = axes[1]
    ax.plot(debug.time, debug.filtered, color='#2ca02c', alpha=0.6, linewidth=1.2, label='Filtered')
    ax.plot(debug.time, debug.median, color='#ff7f0e', alpha=0.9, linewidth=1.6, label='Median baseline')
    ax.set_ylabel('Baseline')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

    ax = axes[2]
    ax.plot(debug.time, debug.detrended, color='#d62728', alpha=0.95, linewidth=1.2, label='Detrended')
    ax.set_ylabel('Detrended')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)
    if algorithm == "fsm":
        ax.plot(debug.time, debug.up_threshold, color='#9467bd', alpha=0.8, linewidth=1.2, linestyle='--', label='Up threshold')
        ax.plot(debug.time, debug.bottom_threshold, color='#9467bd', alpha=0.8, linewidth=1.2, linestyle='--', label='Bottom threshold')

        ax = axes[3]
        # Overlay FSM input (thresholded) and event pins
        ax.step(debug.time, debug.signal_thresholded, where='post', color='#9467bd', linewidth=1.2, label='Thresholded (-1/0/1)')
        pred_enter = np.asarray(fsm_output.enter_ts, dtype=float)
        pred_leave = np.asarray(fsm_output.leave_ts, dtype=float)
        if pred_enter.size > 0:
            ax.vlines(pred_enter, 0, 1, colors='green', linewidth=2, label='Enter')
            ax.scatter(pred_enter, np.ones_like(pred_enter), color='green', s=50, zorder=5)
        if pred_leave.size > 0:
            ax.vlines(pred_leave, -1, 0, colors='red', linewidth=2, label='Leave')
            ax.scatter(pred_leave, -np.ones_like(pred_leave), color='red', s=50, zorder=5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Leave', '', 'Enter'])
        ax.set_ylabel('Events')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=False)

        ax = axes[4]
        ax.plot(debug.time, err_density, color='#ff9896', linewidth=1.5, label='Error density (±10 s)')
        ax.set_ylabel('Errors')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=False)

        save_path = "algorithm-debug.npz"
        np.savez_compressed(
            save_path,
            time=debug.time,
            detrended=debug.detrended,
            up_threshold=debug.up_threshold,
            bottom_threshold=debug.bottom_threshold,
            signal_thresholded=debug.signal_thresholded,
            fsm_enter_ts=fsm_output.enter_ts,
            fsm_leave_ts=fsm_output.leave_ts,
            enter_ts=enter_ts,
            leave_ts=leave_ts
        )
        print(f"Data successfully saved to {save_path}")

    axes[-1].set_xlabel('Time (s)')

    if algorithm == 'correlation':
        ax = axes[3]
        # Overlay FSM input (thresholded) and event pins
        for i, correlation in enumerate(debug.correlations):
            ax.step(debug.time, correlation, label=f"cross correlation: kernel{i}")
            ax.set_ylabel("correlation")
        ax.legend(loc='upper right', frameon=False)

        ax = axes[4]
        ax.step(debug.time, debug.total_correlation, label="avg cross correlation")
        ax.plot(debug.time, debug.up_threshold, color='#9467bd', alpha=0.8, linewidth=1.2, linestyle='--', label='Up threshold')
        ax.plot(debug.time, debug.bottom_threshold, color='#9467bd', alpha=0.8, linewidth=1.2, linestyle='--', label='Bottom threshold')
        pred_enter = np.asarray(fsm_output.enter_ts, dtype=float)
        pred_leave = np.asarray(fsm_output.leave_ts, dtype=float)
        if pred_enter.size > 0:
            ax.vlines(pred_enter, 0, 1, colors='green', linewidth=2, label='Enter')
            ax.scatter(pred_enter, np.ones_like(pred_enter), color='green', s=50, zorder=5)
        if pred_leave.size > 0:
            ax.vlines(pred_leave, -1, 0, colors='red', linewidth=2, label='Leave')
            ax.scatter(pred_leave, -np.ones_like(pred_leave), color='red', s=50, zorder=5)
        ax.set_yticklabels(['Leave', '', 'Enter'])
        ax.set_ylabel('Events')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=False)

        save_path = "algorithm-debug.npz"
        np.savez_compressed(
            save_path,
            time=debug.time,
            detrended=debug.detrended,
            up_threshold=debug.up_threshold,
            bottom_threshold=debug.bottom_threshold,
            correlation_signal=debug.total_correlation,
            fsm_enter_ts=fsm_output.enter_ts,
            fsm_leave_ts=fsm_output.leave_ts,
            enter_ts=enter_ts,
            leave_ts=leave_ts
        )
        print(f"Data successfully saved to {save_path}")


    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
