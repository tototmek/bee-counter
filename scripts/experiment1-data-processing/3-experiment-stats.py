#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def load_all_annotations():
    """Load all tunnel annotation files and combine them into a single DataFrame"""
    annotations_dir = 'data/experiments/manual-annotation'
    all_annotations = []
    
    for i in range(8):  # tunnels 0-7
        tunnel_file = f'{annotations_dir}/tunel{i}.csv'
        if os.path.exists(tunnel_file):
            df = pd.read_csv(tunnel_file)
            df['tunnel'] = i
            all_annotations.append(df)
    
    if not all_annotations:
        raise FileNotFoundError("No annotation files found")
    
    combined_df = pd.concat(all_annotations, ignore_index=True)
    combined_df = combined_df.sort_values('timestamp')
    return combined_df

def calculate_bee_counts_over_time(annotations_df, time_resolution=1.0):
    """Calculate bee counts over time with specified resolution"""
    # Create time bins
    min_time = annotations_df['timestamp'].min()
    max_time = annotations_df['timestamp'].max()
    time_bins = np.arange(min_time, max_time + time_resolution, time_resolution)
    
    # Initialize count arrays
    total_counts = np.zeros(len(time_bins) - 1)
    tunnel_counts = {i: np.zeros(len(time_bins) - 1) for i in range(8)}
    
    # Calculate cumulative counts for each time bin
    for i, (start_time, end_time) in enumerate(zip(time_bins[:-1], time_bins[1:])):
        # Filter events up to this time point
        events_up_to_time = annotations_df[annotations_df['timestamp'] <= end_time]
        
        # Calculate total bees in hive
        enters = len(events_up_to_time[events_up_to_time['event_type'] == 'enter'])
        leaves = len(events_up_to_time[events_up_to_time['event_type'] == 'leave'])
        total_counts[i] = enters - leaves
        
        # Calculate counts per tunnel
        for tunnel in range(8):
            tunnel_events = events_up_to_time[events_up_to_time['tunnel'] == tunnel]
            tunnel_enters = len(tunnel_events[tunnel_events['event_type'] == 'enter'])
            tunnel_leaves = len(tunnel_events[tunnel_events['event_type'] == 'leave'])
            tunnel_counts[tunnel][i] = tunnel_enters - tunnel_leaves
    
    time_points = time_bins[:-1] + time_resolution / 2  # Center of each bin
    return time_points, total_counts, tunnel_counts

def create_statistics_table(annotations_df):
    """Create a summary statistics table"""
    stats = []
    
    # Global statistics
    global_enters = len(annotations_df[annotations_df['event_type'] == 'enter'])
    global_leaves = len(annotations_df[annotations_df['event_type'] == 'leave'])
    global_net = global_enters - global_leaves
    
    stats.append({
        'Tunnel': 'ALL',
        'Bees Entering': global_enters,
        'Bees Leaving': global_leaves,
        'Net Change': global_net
    })
    
    # Per tunnel statistics
    for tunnel in range(8):
        tunnel_data = annotations_df[annotations_df['tunnel'] == tunnel]
        if len(tunnel_data) > 0:
            enters = len(tunnel_data[tunnel_data['event_type'] == 'enter'])
            leaves = len(tunnel_data[tunnel_data['event_type'] == 'leave'])
            net = enters - leaves
        else:
            enters = leaves = net = 0
        
        stats.append({
            'Tunnel': f'Tunnel {tunnel}',
            'Bees Entering': enters,
            'Bees Leaving': leaves,
            'Net Change': net
        })
    
    return pd.DataFrame(stats)

def plot_bee_counts(time_points, total_counts, tunnel_counts):
    """Create comprehensive plots of bee counts over time"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Total bee count over time
    ax1 = axes[0]
    ax1.plot(time_points, total_counts, 'b-', linewidth=2, label='Total Relative Bees in Hive')
    ax1.set_ylabel('Number of Bees')
    ax1.set_title('Total Relative Bee Count in Hive Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add horizontal line at y=0 for reference
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Bee counts per tunnel
    ax2 = axes[1]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for tunnel in range(8):
        ax2.plot(time_points, tunnel_counts[tunnel], 
                color=colors[tunnel], linewidth=1.5, 
                label=f'Tunnel {tunnel}', alpha=1)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Number of Bees')
    ax2.set_title('Bee Count per Tunnel Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def display_statistics_table(stats_df):
    """Display the statistics table in a formatted way"""
    print("\n" + "="*60)
    print("BEE COUNT STATISTICS")
    print("="*60)
    
    # Format the table for display
    formatted_df = stats_df.copy()
    formatted_df['Bees Entering'] = formatted_df['Bees Entering'].astype(int)
    formatted_df['Bees Leaving'] = formatted_df['Bees Leaving'].astype(int)
    formatted_df['Net Change'] = formatted_df['Net Change'].astype(int)
    
    print(formatted_df.to_string(index=False))
    print("="*60)
    
    # Additional insights
    print(f"\nKey Insights:")
    print(f"- Total bees that entered the hive: {stats_df.iloc[0]['Bees Entering']}")
    print(f"- Total bees that left the hive: {stats_df.iloc[0]['Bees Leaving']}")
    print(f"- Net change in hive population: {stats_df.iloc[0]['Net Change']}")
    
    # Find most active tunnels
    tunnel_stats = stats_df.iloc[1:]  # Exclude 'ALL' row
    most_entering = tunnel_stats.loc[tunnel_stats['Bees Entering'].idxmax()]
    most_leaving = tunnel_stats.loc[tunnel_stats['Bees Leaving'].idxmax()]
    
    print(f"- Most active tunnel for entering: {most_entering['Tunnel']} ({most_entering['Bees Entering']} bees)")
    print(f"- Most active tunnel for leaving: {most_leaving['Tunnel']} ({most_leaving['Bees Leaving']} bees)")

def main():
    try:
        print("Loading annotation data...")
        annotations_df = load_all_annotations()
        
        print(f"Loaded {len(annotations_df)} events from {annotations_df['tunnel'].nunique()} tunnels")
        print(f"Time range: {annotations_df['timestamp'].min():.2f} - {annotations_df['timestamp'].max():.2f} seconds")
        
        # Calculate bee counts over time
        print("Calculating bee counts over time...")
        time_points, total_counts, tunnel_counts = calculate_bee_counts_over_time(annotations_df)
        
        # Create statistics table
        print("Creating statistics table...")
        stats_df = create_statistics_table(annotations_df)
        
        # Display statistics
        display_statistics_table(stats_df)
        
        # Create plots
        print("Creating visualizations...")
        fig = plot_bee_counts(time_points, total_counts, tunnel_counts)
        
        # Save the plot
        output_file = 'images/bee_count_analysis.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        
        # Save statistics to CSV
        stats_file = 'images/bee_count_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"Statistics saved to: {stats_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 