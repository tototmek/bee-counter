#!/usr/bin/env python3

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fsm import FsmInput, FsmConfig, run_fsm


RAW_DATA_FILE = 'data/experiments/processed-data/raw-time-adjusted.csv'
ANNOTATION_DIR = 'data/experiments/manual-annotation'


def load_signal(channel_idx: int, start_time: float, end_time: float,
                raw_data_file: str = RAW_DATA_FILE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load channel signal data and corresponding time within a given time range.

    Args:
        channel_idx: Channel index (0-7)
        start_time: Start time in seconds (inclusive)
        end_time: End time in seconds (inclusive)
        raw_data_file: Path to raw data CSV containing 'time' and 'delta{idx}' columns

    Returns:
        (time, signal) as numpy arrays within [start_time, end_time]
    """
    if not os.path.exists(raw_data_file):
        raise FileNotFoundError(f"Raw data not found: {raw_data_file}")

    channel_name = f'delta{channel_idx}'
    df = pd.read_csv(raw_data_file)
    if 'time' not in df.columns or channel_name not in df.columns:
        raise ValueError(f"Raw data must contain 'time' and channel column '{channel_name}'")

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


def parse_args():
    parser = argparse.ArgumentParser(description='FSM data loader: load channel signal and annotation timestamps')
    parser.add_argument('--channel', '-c', type=int, required=True, help='Channel index (0-7)')
    parser.add_argument('--start', '-s', type=float, required=True, help='Start time in seconds')
    parser.add_argument('--end', '-e', type=float, required=True, help='End time in seconds')
    parser.add_argument('--raw', type=str, default=RAW_DATA_FILE, help='Path to raw data CSV')
    parser.add_argument('--ann_dir', type=str, default=ANNOTATION_DIR, help='Path to annotation directory')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.channel < 0 or args.channel > 7:
        print("Error: Channel index must be between 0 and 7")
        return 1
    if args.start >= args.end:
        print("Error: Start time must be less than end time")
        return 1

    try:
        time, signal = load_signal(args.channel, args.start, args.end, args.raw)
        enter_ts, leave_ts = load_annotation_timestamps(args.channel, args.start, args.end, args.ann_dir)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    print(f"Loaded signal samples: {signal.size}")
    print(f"Time samples: {time.size}")
    print(f"Enter events in range: {enter_ts.size}")
    print(f"Leave events in range: {leave_ts.size}")

    fsm_input = FsmInput(time, signal)
    fsm_output, debug = run_fsm(fsm_input, FsmConfig())

    # Plot debug outputs
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle('FSM Preprocessing', fontsize=14)

    ax = axes[0]
    ax.plot(debug.time, debug.raw, color='#1f77b4', alpha=0.35, linewidth=0.8, label='Raw')
    ax.plot(debug.time, debug.filtered, color='#1f77b4', alpha=0.95, linewidth=1.6, label='Filtered (MA)')
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
    ax.axhline(0.0, color='k', linewidth=1, alpha=0.5, linestyle='--')
    thr = debug.threshold
    ax.axhline(+thr, color='#9467bd', linewidth=1, alpha=0.8, linestyle=':')
    ax.axhline(-thr, color='#9467bd', linewidth=1, alpha=0.8, linestyle=':')
    ax.set_ylabel('Detrended')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=False)

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

    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
