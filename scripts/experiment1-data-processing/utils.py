import numpy as np
import pandas as pd
import os

def load_data(file_path):
    """
    Load data from CSV files in the experiments directory.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: DataFrame with columns 'time', 'delta0', 'delta1', ..., 'delta7'
    
    Example:
        df = load_data("2025-07-20/2025-07-20_12-25-59.csv")
        
        # Access data
        time_data = df['time']
        delta0_data = df['delta0']
        # Or access multiple columns
        sensor_data = df[['delta0', 'delta1', 'delta2', 'delta3']]
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Read CSV file using pandas and return DataFrame directly
        df = pd.read_csv(file_path)
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file {file_path}: {str(e)}")