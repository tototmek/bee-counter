#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def visualize_annotations(channel_idx, start_time, end_time):
    # Load data
    raw_data = pd.read_csv('data/experiments/processed-data/raw-time-adjusted.csv')
    annotation_file = f'data/experiments/manual-annotation/tunel{channel_idx}.csv'
    annotations = pd.read_csv(annotation_file)
    
    # Filter data by time range
    raw_mask = (raw_data['time'] >= start_time) & (raw_data['time'] <= end_time)
    raw_filtered = raw_data[raw_mask]
    
    annotation_mask = (annotations['timestamp'] >= start_time) & (annotations['timestamp'] <= end_time)
    annotations_filtered = annotations[annotation_mask]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot raw data
    channel_name = f'delta{channel_idx}'
    ax1.plot(raw_filtered['time'], raw_filtered[channel_name], 'b-', linewidth=0.5, alpha=0.3, label='Raw Data')
    
    # Apply rolling average filter with window size 30
    filtered_data = raw_filtered[channel_name].rolling(window=30, center=True).mean()
    ax1.plot(raw_filtered['time'], filtered_data, 'b-', linewidth=1.5, alpha=0.8, label='Filtered data')
    
    # Scale the plot according to filtered data range
    filtered_min = filtered_data.min()
    filtered_max = filtered_data.max()
    margin = (filtered_max - filtered_min) * 0.1  # Add 10% margin
    ax1.set_ylim(filtered_min - margin, filtered_max + margin)
    
    ax1.set_ylabel(f'Data ({channel_name})')
    ax1.set_title(f'Channel {channel_idx} Data Visualization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot annotations
    enter_times = annotations_filtered[annotations_filtered['event_type'] == 'enter']['timestamp']
    leave_times = annotations_filtered[annotations_filtered['event_type'] == 'leave']['timestamp']
    
    # Plot enter events as +1 with vertical lines
    if not enter_times.empty:
        ax2.vlines(enter_times, 0, 1, colors='green', linewidth=2, label='Enter')
        ax2.scatter(enter_times, [1] * len(enter_times), color='green', marker='o', s=50, zorder=5)
    
    # Plot leave events as -1 with vertical lines
    if not leave_times.empty:
        ax2.vlines(leave_times, -1, 0, colors='red', linewidth=2, label='Leave')
        ax2.scatter(leave_times, [-1] * len(leave_times), color='red', marker='o', s=50, zorder=5)
    
    ax2.set_ylabel('Annotations')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Leave', '', 'Enter'])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Set x-axis limits
    ax1.set_xlim(start_time, end_time)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize raw data with manual annotations')
    parser.add_argument('channel', type=int, help='Channel index (0-7)')
    parser.add_argument('start_time', type=float, help='Start time in seconds')
    parser.add_argument('end_time', type=float, help='End time in seconds')
    
    args = parser.parse_args()
    
    if args.channel < 0 or args.channel > 7:
        print("Error: Channel index must be between 0 and 7")
        return 1
    
    if args.start_time >= args.end_time:
        print("Error: Start time must be less than end time")
        return 1
    
    try:
        visualize_annotations(args.channel, args.start_time, args.end_time)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 