#!/usr/bin/env python3

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fsm import moving_average, moving_median

RAW_DATA_FILE = 'data/experiments/processed-data/raw-time-adjusted.csv'


def load_signal(raw_path: str, channel_idx: int, start: Optional[float], end: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    ch = f'delta{channel_idx}'
    df = pd.read_csv(raw_path)
    if 'time' not in df.columns or ch not in df.columns:
        raise ValueError(f"CSV must contain 'time' and '{ch}' columns")
    if start is None or end is None:
        mask = np.ones(len(df), dtype=bool)
    else:
        if start >= end:
            raise ValueError('start must be less than end')
        mask = (df['time'] >= start) & (df['time'] <= end)
    return df.loc[mask, 'time'].to_numpy(dtype=float), df.loc[mask, ch].to_numpy(dtype=float)


def parse_args():
    p = argparse.ArgumentParser(description='Plot normalized histogram of detrended signal values')
    p.add_argument('--channel', '-c', type=int, required=True, help='Channel index (0-7)')
    p.add_argument('--start', '-s', type=float, default=None, help='Start time (s), default full')
    p.add_argument('--end', '-e', type=float, default=None, help='End time (s), default full')
    p.add_argument('--raw', type=str, default=RAW_DATA_FILE, help='Path to raw CSV')
    p.add_argument('--bins', type=int, default=100, help='Number of histogram bins (default: 100)')
    p.add_argument('--filter-window', type=int, default=100, help='Moving average window length (samples)')
    p.add_argument('--detrend-window', type=int, default=200, help='Moving median window length (samples)')
    p.add_argument('--tail-q', type=float, default=0.1, help='Tail quantile (e.g., 0.1 marks bottom/top 10%)')
    p.add_argument('--mult', type=float, default=1.25, help='Multiplier for threshold lines (e.g., 1.25)')
    p.add_argument('--save', type=str, default=None, help='Optional path to save the figure (PNG)')
    p.add_argument('--no-show', action='store_true', help='Do not show figure')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.channel < 0 or args.channel > 7:
        print('Error: Channel index must be between 0 and 7')
        return 1

    try:
        time, values = load_signal(args.raw, args.channel, args.start, args.end)
        if values.size == 0:
            print('Error: No samples in selected range')
            return 1
        if args.filter_window <= 0 or args.detrend_window <= 0:
            print('Error: filter-window and detrend-window must be positive integers')
            return 1
        q = float(args.tail_q)
        if not (0.0 < q < 0.5):
            print('Error: --tail-q must be in (0, 0.5)')
            return 1
        mult = float(args.mult)
    except Exception as e:
        print(f'Error: {e}')
        return 1

    # Filter and detrend like in run_fsm
    filtered = moving_average(values, args.filter_window)
    baseline = moving_median(filtered, args.detrend_window)
    detrended = filtered - baseline

    # Quantile thresholds from detrended data
    q_low = float(np.quantile(detrended, q))
    q_high = float(np.quantile(detrended, 1.0 - q))
    q_low_scaled = q_low * mult
    q_high_scaled = q_high * mult

    fig, (ax_hist, ax_sig) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Top: histogram of detrended values
    ax_hist.hist(detrended, bins=args.bins, density=True, color='#1f77b4', alpha=0.85, edgecolor='white')
    # Vertical lines at quantiles
    ax_hist.axvline(q_low, color='#2ca02c', linestyle='--', linewidth=1.6, label=f'low q={q:.2f}')
    ax_hist.axvline(q_high, color='#d62728', linestyle='--', linewidth=1.6, label=f'high q={1.0 - q:.2f}')
    ax_hist.axvline(q_low_scaled, color='#2ca02c', linestyle=':', linewidth=1.4, alpha=0.9, label=f'low×{mult:.2f}')
    ax_hist.axvline(q_high_scaled, color='#d62728', linestyle=':', linewidth=1.4, alpha=0.9, label=f'high×{mult:.2f}')
    title_range = 'full range' if args.start is None or args.end is None else f'{args.start:.2f}s - {args.end:.2f}s'
    ax_hist.set_title(f'Channel delta{args.channel} detrended signal histogram ({title_range})')
    ax_hist.set_xlabel('Detrended signal value')
    ax_hist.set_ylabel('Density')
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend(loc='upper right', frameon=False)

    # Bottom: detrended signal over time
    ax_sig.plot(time, detrended, color='#d62728', linewidth=1.0, alpha=0.95, label='Detrended')
    # Horizontal lines at quantiles and scaled thresholds
    ax_sig.axhline(q_low, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.9, label='low q')
    ax_sig.axhline(q_high, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.9, label='high q')
    ax_sig.axhline(q_low_scaled, color='#2ca02c', linestyle=':', linewidth=1.0, alpha=0.9, label='low×mult')
    ax_sig.axhline(q_high_scaled, color='#d62728', linestyle=':', linewidth=1.0, alpha=0.9, label='high×mult')
    ax_sig.axhline(0.0, color='k', linewidth=1, alpha=0.4, linestyle='--')
    ax_sig.set_xlabel('Time (s)')
    ax_sig.set_ylabel('Detrended')
    ax_sig.grid(True, alpha=0.3)
    ax_sig.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    if args.save:
        try:
            os.makedirs(os.path.dirname(args.save), exist_ok=True) if os.path.dirname(args.save) else None
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f'Saved figure to {args.save}')
        except Exception as e:
            print(f'Warning: could not save figure: {e}')
    if not args.no_show:
        plt.show()

    return 0


if __name__ == '__main__':
    raise SystemExit(main()) 