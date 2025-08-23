#!/usr/bin/env python3

import argparse
import json
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import optuna
from optuna.samplers import TPESampler

from fsm import FsmInput, FsmConfig, run_fsm


RAW_DATA_FILE = 'data/experiments/processed-data/raw-time-adjusted.csv'
ANNOTATION_DIR = 'data/experiments/manual-annotation'


def load_raw_dataframe(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)
    if 'time' not in df.columns:
        raise ValueError("Raw data must contain 'time' column")
    return df


def load_annotations_for_channel(channel_idx: int, ann_dir: str) -> pd.DataFrame:
    ann_file = os.path.join(ann_dir, f'tunel{channel_idx}.csv')
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    df = pd.read_csv(ann_file)
    if not {'timestamp', 'event_type'}.issubset(df.columns):
        raise ValueError("Annotation CSV must contain 'timestamp' and 'event_type'")
    return df


def slice_ranges(min_t: float, max_t: float, start: Optional[float], end: Optional[float],
                 num_slices: int, slice_length: Optional[float]) -> List[Tuple[float, float]]:
    # Effective range
    s = min_t if start is None else max(min_t, start)
    e = max_t if end is None else min(max_t, end)
    if s >= e:
        return []
    # Single full slice
    if slice_length is None or slice_length <= 0 or slice_length >= (e - s):
        return [(s, e)]
    # Create evenly spaced slice starts
    if num_slices <= 1:
        return [(s, min(e, s + slice_length))]
    span = e - s
    step = max(1e-6, (span - slice_length) / (num_slices - 1))
    slices: List[Tuple[float, float]] = []
    for i in range(num_slices):
        st = s + i * step
        en = min(e, st + slice_length)
        if en - st <= 0:
            continue
        slices.append((st, en))
    return slices


def extract_time_signal(raw_df: pd.DataFrame, channel_idx: int, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    ch = f'delta{channel_idx}'
    if ch not in raw_df.columns:
        raise ValueError(f"Raw data must contain channel column '{ch}'")
    m = (raw_df['time'] >= t0) & (raw_df['time'] <= t1)
    times = raw_df.loc[m, 'time'].to_numpy(dtype=float)
    sig = raw_df.loc[m, ch].to_numpy(dtype=float)
    return times, sig


def extract_ann_timestamps(ann_df: pd.DataFrame, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    m = (ann_df['timestamp'] >= t0) & (ann_df['timestamp'] <= t1)
    d = ann_df.loc[m]
    enters = d[d['event_type'] == 'enter']['timestamp'].to_numpy(dtype=float)
    leaves = d[d['event_type'] == 'leave']['timestamp'].to_numpy(dtype=float)
    return enters, leaves


def compute_event_metrics(gt_ts: np.ndarray, pred_ts: np.ndarray, tol_s: float) -> Tuple[int, int, int]:
    gt = np.sort(gt_ts.astype(float))
    pred = np.sort(np.asarray(pred_ts, dtype=float))
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
        else:
            fn += 1
            i += 1
    fp += max(0, len(pred) - j)
    fn += max(0, len(gt) - i)
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def build_objective(raw_df: pd.DataFrame, channels: List[int], ann_by_channel: Dict[int, pd.DataFrame],
                    ranges: List[Tuple[float, float]], tol_s: float):
    def objective(trial: optuna.trial.Trial) -> float:
        # Search space
        filter_window = trial.suggest_int('filter_window', 10, 400)
        detrend_window = trial.suggest_int('detrend_window', filter_window + 10, 1500)
        timeout_samples = trial.suggest_int('timeout_samples', 20, 1500)
        input_threshold = trial.suggest_float('input_threshold', 0.05, 1.0)

        # Aggregate counts across all channels and slices (micro-average)
        tp_e = fp_e = fn_e = 0
        tp_l = fp_l = fn_l = 0

        cfg = FsmConfig()
        cfg.filter_window = filter_window
        cfg.detrend_window = detrend_window
        cfg.timeout_samples = timeout_samples
        cfg.input_threshold = input_threshold

        for ch in channels:
            ann_df = ann_by_channel[ch]
            for (t0, t1) in ranges:
                time, signal = extract_time_signal(raw_df, ch, t0, t1)
                if time.size == 0:
                    continue
                gt_enter, gt_leave = extract_ann_timestamps(ann_df, t0, t1)
                fsm_input = FsmInput(time, signal)
                fsm_output, _ = run_fsm(fsm_input, cfg)
                pred_enter = np.asarray(fsm_output.enter_ts, dtype=float)
                pred_leave = np.asarray(fsm_output.leave_ts, dtype=float)
                te, fe, ne = compute_event_metrics(gt_enter, pred_enter, tol_s)
                tl, fl, nl = compute_event_metrics(gt_leave, pred_leave, tol_s)
                tp_e += te; fp_e += fe; fn_e += ne
                tp_l += tl; fp_l += fl; fn_l += nl

        f1_e = f1_from_counts(tp_e, fp_e, fn_e)
        f1_l = f1_from_counts(tp_l, fp_l, fn_l)
        score = 0.5 * (f1_e + f1_l)
        return score

    return objective


def parse_args():
    p = argparse.ArgumentParser(description='Optimize FSM parameters with Optuna')
    p.add_argument('--channels', type=str, default='0', help='Comma-separated channel indices (e.g., 0,1,2)')
    p.add_argument('--start', type=float, default=None, help='Start time (s), default full')
    p.add_argument('--end', type=float, default=None, help='End time (s), default full')
    p.add_argument('--num-slices', type=int, default=8, help='Number of slices across range')
    p.add_argument('--slice-length', type=float, default=300.0, help='Slice length in seconds (<=range => single full slice)')
    p.add_argument('--tol', type=float, default=3.0, help='Tolerance (s) for matching events')
    p.add_argument('--raw', type=str, default=RAW_DATA_FILE, help='Path to raw CSV')
    p.add_argument('--ann-dir', type=str, default=ANNOTATION_DIR, help='Path to annotation directory')
    p.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    p.add_argument('--seed', type=int, default=42, help='Random seed for the sampler')
    p.add_argument('--study-name', type=str, default='fsm-param-optimization', help='Optuna study name')
    p.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    p.add_argument('--out', type=str, default='best_fsm_params.json', help='Where to save best params JSON')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        channels = [int(c) for c in args.channels.split(',') if c.strip() != '']
    except Exception:
        print('Error: --channels must be a comma-separated list of integers, e.g., "0,1,2"')
        return 1
    if len(channels) == 0:
        print('Error: No channels specified')
        return 1

    try:
        raw_df = load_raw_dataframe(args.raw)
        min_t = float(raw_df['time'].min())
        max_t = float(raw_df['time'].max())
        ranges = slice_ranges(min_t, max_t, args.start, args.end, args.num_slices, args.slice_length)
        if not ranges:
            print('Error: Computed range is empty')
            return 1
        ann_by_channel: Dict[int, pd.DataFrame] = {ch: load_annotations_for_channel(ch, args.ann_dir) for ch in channels}
    except Exception as e:
        print(f'Error setting up data: {e}')
        return 1

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=args.study_name,
                                storage=args.storage, load_if_exists=(args.storage is not None))

    objective = build_objective(raw_df, channels, ann_by_channel, ranges, args.tol)

    try:
        study.optimize(objective, n_trials=args.n_trials)
    except Exception as e:
        print(f'Error during optimization: {e}')
        return 1

    best = study.best_trial
    print('\nBest score: %.4f' % best.value)
    print('Best params:')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # Save best params JSON
    try:
        with open(args.out, 'w') as f:
            json.dump({'score': best.value, 'params': best.params}, f, indent=2)
        print(f'Saved best params to: {args.out}')
    except Exception as e:
        print(f'Warning: could not save best params: {e}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
