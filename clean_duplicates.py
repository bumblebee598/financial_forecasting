import os
import re
from datetime import datetime
import pandas as pd

def clean_duplicate_stock_files(directory='./stock_data'):
    """
    Identifies and removes duplicate stock data files, keeping only the most recent file for each stock symbol.
    
    Parameters:
    -----------
    directory: str
        Path to the directory containing stock data files
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return
    
    # Get all CSV files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not files:
        print(f"No CSV files found in {directory}.")
        return
    
    # Group files by stock symbol
    stock_files = {}
    for file in files:
        # Extract stock symbol and timestamp using regex
        match = re.match(r'(.+)_monthly_adjusted_(\d{8}_\d{6})\.csv', file)
        if match:
            symbol = match.group(1)
            timestamp_str = match.group(2)
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            
            if symbol not in stock_files:
                stock_files[symbol] = []
            
            stock_files[symbol].append((file, timestamp))
    
    # Count files before deletion
    total_files = len(files)
    deleted_files = 0
    
    # For each stock symbol, keep only the most recent file
    for symbol, file_list in stock_files.items():
        if len(file_list) > 1:
            # Sort files by timestamp (newest first)
            sorted_files = sorted(file_list, key=lambda x: x[1], reverse=True)
            
            # Keep the most recent file, delete the rest
            print(f"For {symbol}, keeping {sorted_files[0][0]} (newest)")
            
            for file, _ in sorted_files[1:]:
                file_path = os.path.join(directory, file)
                try:
                    os.remove(file_path)
                    print(f"  Deleted: {file}")
                    deleted_files += 1
                except Exception as e:
                    print(f"  Error deleting {file}: {e}")
    
    print(f"\nSummary: Deleted {deleted_files} duplicate files out of {total_files} total files.")
    print(f"Remaining files: {total_files - deleted_files}")

if __name__ == "__main__":
    clean_duplicate_stock_files() 