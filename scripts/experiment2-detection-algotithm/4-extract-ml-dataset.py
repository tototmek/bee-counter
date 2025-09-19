import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from algorithm import moving_average, moving_median

window_length = 300
window_overlap = 60
filter_window = 32
detrend_window = 1500

RAW_DATA_FILE = 'data/experiments/processed-data/raw-time-adjusted.csv'
ANNOTATION_DIR = 'data/experiments/manual-annotation'

def load_signal(channel_idx: int, raw_data_file: str = RAW_DATA_FILE):
    channel_name = f'delta{channel_idx}'
    df = pd.read_csv(raw_data_file)
    if 'time' not in df.columns or channel_name not in df.columns:
        raise ValueError(f"Raw data must contain 'time' and channel column '{channel_name}'")

    mask = np.ones(len(df), dtype=bool)

    times = df.loc[mask, 'time'].to_numpy(dtype=float)
    values = df.loc[mask, channel_name].to_numpy(dtype=float)
    return times, values


def load_annotation_timestamps(channel_idx: int, annotation_dir: str = ANNOTATION_DIR):
    ann_file = os.path.join(annotation_dir, f'tunel{channel_idx}.csv')
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    ann = pd.read_csv(ann_file)
    if not {'timestamp', 'event_type'}.issubset(ann.columns):
        raise ValueError("Annotation CSV must contain 'timestamp' and 'event_type'")

    enters = ann[ann['event_type'] == 'enter']['timestamp'].to_numpy(dtype=float)
    leaves = ann[ann['event_type'] == 'leave']['timestamp'].to_numpy(dtype=float)
    return enters, leaves


# Load the raw data
time, data = load_signal(0)
annotation = {}
annotation["enter"], annotation["leave"] = load_annotation_timestamps(0)

# Detrend the signal
data = moving_average(data, filter_window)
median = moving_median(data, detrend_window)
data = data - median

# Split into overlapping chunks
dataset = []
index = 0
chunk_bounds = []
while index < (len(data) - window_length):
    dataset.append(data[index:index+window_length])
    chunk_bounds.append((time[index], time[min(len(time)-1, index+window_length)]))
    index += window_length - window_overlap

annotation_enter = [0] * len(dataset)
annotation_leave = [0] * len(dataset)

for chunk_idx, chunk_bounds in enumerate(chunk_bounds):
    for ann in annotation["enter"]:
        if ann > chunk_bounds[0] and ann <= chunk_bounds[1]:
            annotation_enter[chunk_idx] = 1
    for ann in annotation["leave"]:
        if ann > chunk_bounds[0] and ann <= chunk_bounds[1]:
            annotation_leave[chunk_idx] = 1

dataset = np.array(dataset)
annotation_enter = np.array(annotation_enter)
annotation_leave = np.array(annotation_leave)
np.save("scripts/experiment2-detection-algotithm/data/signal_dataset.npy", dataset)
np.save("scripts/experiment2-detection-algotithm/data/annotation_enter.npy", annotation_enter)
np.save("scripts/experiment2-detection-algotithm/data/annotation_leave.npy", annotation_leave)


# N = 10
# fig, axes = plt.subplots(N, 1, figsize=(14, 13), sharex=True)

# enter_chunks = []
# for i in range(len(dataset)):
#     if annotation_enter[i] == 1:
#         enter_chunks.append(dataset[i])
# print(len(enter_chunks))
# for i in range(N):
#     ax = axes[i]
#     ax.plot(enter_chunks[i])
# plt.show()