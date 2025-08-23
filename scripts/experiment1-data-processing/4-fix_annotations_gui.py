#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


@dataclass
class Event:
    timestamp: float
    event_type: str  # 'enter' or 'leave'

    def to_tuple(self) -> Tuple[float, str]:
        return (self.timestamp, self.event_type)


class AnnotationEditor:
    def __init__(self, channel_idx: int, window_s: float):
        self.channel_idx = channel_idx
        self.window_s = window_s
        self.channel_name = f"delta{channel_idx}"

        # Files
        self.raw_data_file = 'data/experiments/processed-data/raw-time-adjusted.csv'
        self.annotation_file = f'data/experiments/manual-annotation/tunel{channel_idx}.csv'
        self.output_file = f'data/experiments/manual-annotation/tunel{channel_idx}_edited.csv'

        # Data
        self.raw_df = None  # type: Optional[pd.DataFrame]
        self.events: List[Event] = []

        # View state
        self.t_min = 0.0
        self.t_max = 0.0
        self.view_start = 0.0
        self.view_end = 0.0

        # Matplotlib state
        self.fig = None
        self.ax_signal = None
        self.line_signal = None
        self.line_signal_filtered = None
        self.event_artists = []  # list of (line, scatter, type)
        self.selected_idx: Optional[int] = None
        self.dragging = False
        self.drag_start_x = None

        # Interaction config
        self.pick_tolerance = 5  # pixels
        self.filter_window = 30  # samples for rolling mean

    # ---------------------- Data Loading ----------------------
    def load(self) -> None:
        if not os.path.exists(self.raw_data_file):
            raise FileNotFoundError(f"Raw data not found: {self.raw_data_file}")
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        self.raw_df = pd.read_csv(self.raw_data_file)
        if 'time' not in self.raw_df.columns or self.channel_name not in self.raw_df.columns:
            raise ValueError(f"Raw data must contain 'time' and channel column '{self.channel_name}'")

        ann_df = pd.read_csv(self.annotation_file)
        if not {'timestamp', 'event_type'}.issubset(ann_df.columns):
            raise ValueError("Annotation CSV must contain 'timestamp' and 'event_type'")
        # Normalize types
        ann_df = ann_df.dropna(subset=['timestamp', 'event_type'])
        ann_df['timestamp'] = ann_df['timestamp'].astype(float)
        ann_df['event_type'] = ann_df['event_type'].astype(str)
        # Load events as list of dataclasses, sorted by time
        self.events = [Event(row.timestamp, row.event_type) for row in ann_df.itertuples(index=False)]
        self.events.sort(key=lambda e: e.timestamp)

        # Time range
        self.t_min = float(self.raw_df['time'].min())
        self.t_max = float(self.raw_df['time'].max())
        self.view_start = self.t_min
        self.view_end = min(self.t_min + self.window_s, self.t_max)

    # ---------------------- Rendering ----------------------
    def setup_plot(self) -> None:
        mpl.rcParams['toolbar'] = 'toolmanager'
        self.fig, self.ax_signal = plt.subplots(1, 1, figsize=(12, 6), sharex=False)
        self.fig.canvas.manager.set_window_title(f"Annotation Editor - Channel {self.channel_idx}")

        # Plot initial data slice
        self.refresh_plot(full=True)

        # Connect events
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Add help text
        self.fig.text(0.01, 0.01,
            "Keys: [Left/Right]=prev/next window, a=add enter, s=add leave, del=delete, Ctrl+S=save",
            fontsize=9, alpha=0.7)

    def refresh_plot(self, full: bool = False) -> None:
        # Data slice
        mask = (self.raw_df['time'] >= self.view_start) & (self.raw_df['time'] <= self.view_end)
        slice_df = self.raw_df.loc[mask]

        if full:
            self.ax_signal.clear()

        # Plot signal
        if full or self.line_signal is None:
            self.line_signal, = self.ax_signal.plot(slice_df['time'], slice_df[self.channel_name], 'b-', linewidth=0.5, alpha=0.3, label='Raw Data')
        else:
            self.line_signal.set_data(slice_df['time'].values, slice_df[self.channel_name].values)

        # Filtered signal
        filtered = slice_df[self.channel_name].rolling(window=self.filter_window, center=True).mean()
        if full or self.line_signal_filtered is None:
            self.line_signal_filtered, = self.ax_signal.plot(slice_df['time'], filtered, 'b-', linewidth=1.5, alpha=0.9, label='Filtered')
        else:
            self.line_signal_filtered.set_data(slice_df['time'].values, filtered.values)

        # Scale Y to filtered
        if len(filtered.dropna()) > 0:
            fmin = float(filtered.min())
            fmax = float(filtered.max())
            margin = (fmax - fmin) * 0.1 if fmax > fmin else 1.0
            self.ax_signal.set_ylim(fmin - margin, fmax + margin)
        self.ax_signal.set_ylabel(f"Data ({self.channel_name})")
        self.ax_signal.set_title(f"Channel {self.channel_idx} | {self.view_start:.2f}s - {self.view_end:.2f}s")
        self.ax_signal.grid(True, alpha=0.3)
        self.ax_signal.legend(loc='upper right')

        # Remove previous event artists
        for artists in self.event_artists:
            for art in artists[:2]:
                art.remove()
        self.event_artists = []

        # Plot events on the same axis using axes-fraction for Y
        xform = self.ax_signal.get_xaxis_transform()  # data in X, axes in Y
        window_events = [(i, ev) for i, ev in enumerate(self.events) if self.view_start <= ev.timestamp <= self.view_end]
        enters = [ev.timestamp for _, ev in window_events if ev.event_type == 'enter']
        leaves = [ev.timestamp for _, ev in window_events if ev.event_type == 'leave']

        if enters:
            l1 = self.ax_signal.vlines(enters, 0, 1, transform=xform, colors='green', linewidth=2, label='Enter')
            s1 = self.ax_signal.scatter(enters, [0.9]*len(enters), transform=xform, color='green', s=50, zorder=5, picker=self.pick_tolerance)
            self.event_artists.append((l1, s1, 'enter'))
        if leaves:
            l2 = self.ax_signal.vlines(leaves, 0, 1, transform=xform, colors='red', linewidth=2, label='Leave')
            s2 = self.ax_signal.scatter(leaves, [0.1]*len(leaves), transform=xform, color='red', s=50, zorder=5, picker=self.pick_tolerance)
            self.event_artists.append((l2, s2, 'leave'))

        # X limits
        self.ax_signal.set_xlim(self.view_start, self.view_end)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        self.fig.canvas.draw_idle()

    # ---------------------- Helpers ----------------------
    def find_nearest_event_index(self, x_time: float, event_type: Optional[str] = None, max_dt: float = 1.0) -> Optional[int]:
        # Consider only events within current window +/- tolerance
        candidates = [i for i, ev in enumerate(self.events)
                      if (event_type is None or ev.event_type == event_type)
                      and (self.view_start - max_dt) <= ev.timestamp <= (self.view_end + max_dt)]
        if not candidates:
            return None
        idx = min(candidates, key=lambda i: abs(self.events[i].timestamp - x_time))
        return idx

    def clamp_time(self, t: float) -> float:
        return float(max(self.t_min, min(self.t_max, t)))

    def set_window_centered_on(self, center_time: float) -> None:
        half = self.window_s / 2.0
        start = max(self.t_min, center_time - half)
        end = min(self.t_max, start + self.window_s)
        start = max(self.t_min, end - self.window_s)
        self.view_start, self.view_end = start, end

    # ---------------------- Interaction ----------------------
    def on_pick(self, event):
        # Select nearest event based on scatter pick
        mouse_x = event.mouseevent.xdata
        if mouse_x is None:
            return
        nearest = self.find_nearest_event_index(mouse_x)
        self.selected_idx = nearest
        self.fig.canvas.manager.set_window_title(self._title_with_selection())

    def on_mouse_press(self, event):
        if event.inaxes is not self.ax_signal:
            return
        if event.button == 1:  # Left click
            # Start drag if near selected event
            if self.selected_idx is not None:
                self.dragging = True
                self.drag_start_x = event.xdata
        elif event.button == 3:  # Right click recenters window
            if event.xdata is not None:
                self.set_window_centered_on(event.xdata)
                self.refresh_plot(full=False)

    def on_mouse_release(self, event):
        if event.button == 1 and self.dragging:
            self.dragging = False
            self.drag_start_x = None
            self.refresh_plot(full=False)

    def on_mouse_move(self, event):
        if not self.dragging or self.selected_idx is None:
            return
        if event.xdata is None:
            return
        # Move selected event horizontally
        new_t = self.clamp_time(float(event.xdata))
        self.events[self.selected_idx].timestamp = new_t
        self.events.sort(key=lambda e: e.timestamp)
        self.refresh_plot(full=False)

    def on_key_press(self, event):
        if event.key in ['right', 'pagedown']:
            self.next_window()
        elif event.key in ['left', 'pageup']:
            self.prev_window()
        elif event.key == 'delete':
            self.delete_selected()
        elif event.key == 'a':
            self.add_event('enter')
        elif event.key == 's':
            self.add_event('leave')
        elif event.key == 'ctrl+s':
            self.save()

    def next_window(self):
        shift = self.window_s
        new_start = min(self.t_max - self.window_s, self.view_start + shift)
        new_end = new_start + self.window_s
        if new_end > self.t_max:
            new_end = self.t_max
            new_start = max(self.t_min, new_end - self.window_s)
        self.view_start, self.view_end = new_start, new_end
        self.selected_idx = None
        self.refresh_plot(full=False)

    def prev_window(self):
        shift = self.window_s
        new_start = max(self.t_min, self.view_start - shift)
        new_end = min(self.t_max, new_start + self.window_s)
        self.view_start, self.view_end = new_start, new_end
        self.selected_idx = None
        self.refresh_plot(full=False)

    def delete_selected(self):
        if self.selected_idx is None:
            return
        del self.events[self.selected_idx]
        self.selected_idx = None
        self.refresh_plot(full=False)

    def add_event(self, event_type: str):
        # Add at center of view if mouse not on axis
        mouse_event = plt.ginput(1, timeout=0.1)
        if mouse_event:
            x, _ = mouse_event[0]
        else:
            x = (self.view_start + self.view_end) / 2.0
        x = self.clamp_time(float(x))
        self.events.append(Event(x, event_type))
        self.events.sort(key=lambda e: e.timestamp)
        self.refresh_plot(full=False)

    def save(self):
        out_dir = os.path.dirname(self.output_file)
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame([e.to_tuple() for e in self.events], columns=['timestamp', 'event_type'])
        df.to_csv(self.output_file, index=False)
        self.fig.canvas.manager.set_window_title(self._title_with_selection(saved=True))

    def _title_with_selection(self, saved: bool = False) -> str:
        base = f"Annotation Editor - Channel {self.channel_idx}"
        if self.selected_idx is not None:
            ev = self.events[self.selected_idx]
            base += f" | Selected: {ev.event_type} @ {ev.timestamp:.2f}s"
        if saved:
            base += " | Saved"
        return base


# ---------------------- CLI ----------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Edit manual annotations overlaid on raw data')
    parser.add_argument('--channel', '-c', type=int, required=True, help='Channel index (0-7)')
    parser.add_argument('--window', '-w', type=float, default=100.0, help='Window length in seconds (default: 100)')
    parser.add_argument('--start', '-s', type=float, default=None, help='Optional start time in seconds')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.channel < 0 or args.channel > 7:
        print("Error: Channel index must be between 0 and 7")
        return 1
    if args.window <= 0:
        print("Error: Window length must be positive")
        return 1

    editor = AnnotationEditor(channel_idx=args.channel, window_s=args.window)
    editor.load()
    if args.start is not None:
        editor.view_start = max(editor.t_min, min(float(args.start), editor.t_max))
        editor.view_end = min(editor.t_max, editor.view_start + editor.window_s)
        editor.view_start = max(editor.t_min, editor.view_end - editor.window_s)
    editor.setup_plot()
    plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
