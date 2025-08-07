# Check data size
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_data_size():
    """Check the size of the data without full model training"""
    
    print("Checking data size...")
    
    try:
        # Load data
        df = pd.read_csv('JROAD tokyo data/merged_ami_weather_data.csv')
        print(f"Total data points: {len(df)}")
        print(f"Date range: {df['hospitalization_date'].min()} to {df['hospitalization_date'].max()}")
        
        # Calculate years
        start_date = pd.to_datetime(df['hospitalization_date'].min())
        end_date = pd.to_datetime(df['hospitalization_date'].max())
        years = (end_date - start_date).days / 365.25
        print(f"Data spans approximately {years:.1f} years")
        
        # Check test data size
        test_size = 30  # 1 month
        n_splits = 3
        total_test_days = test_size * n_splits
        print(f"Test data: {total_test_days} days ({total_test_days/365.25:.1f} years)")
        
        # Calculate percentage
        percentage = (total_test_days / len(df)) * 100
        print(f"Test data percentage: {percentage:.1f}%")
        
        return len(df)
        
    except FileNotFoundError:
        print("Data file not found. Checking other possible locations...")
        
        # Try to find data files
        import os
        possible_files = [
            'JROAD tokyo data/merged_ami_weather_data.csv',
            '../JROAD tokyo data/merged_ami_weather_data.csv',
            'merged_ami_weather_data.csv'
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Found data file: {file_path}")
                print(f"Total data points: {len(df)}")
                return len(df)
        
        print("No data files found.")
        return None

if __name__ == "__main__":
    data_size = check_data_size() 