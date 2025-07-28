#!/usr/bin/env python3

import pandas as pd
import argparse

def adjust_timestamps(input_file, output_file, offset=0.0):
    df = pd.read_csv(input_file)
    
    if 'time' not in df.columns:
        raise ValueError("No 'time' column found in the file")
    
    start_time = df['time'].iloc[0]
    df['time'] = (df['time'] - start_time) / 1000.0 + offset
    
    df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Adjust timestamps in CSV files')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--offset', type=float, default=0.0, help='Time offset in seconds')
    
    args = parser.parse_args()
    
    try:
        adjust_timestamps(args.input_file, args.output_file, args.offset)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 