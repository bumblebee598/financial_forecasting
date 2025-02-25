import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

ALPHA_API_KEY = "022WWI8CMAV1LYO9"

def get_monthly_adjusted_stock_data(symbol, output_size='full', output_format='pandas', start_date=None, end_date=None):
    """
    Fetches monthly adjusted time series data from Alpha Vantage API.
    
    Parameters:
    -----------
    api_key: str
        Your Alpha Vantage API key
    symbol: str
        Stock symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    output_size: str
        'full' for up to 20 years of data, 'compact' for last 100 data points (default: 'full')
    output_format: str
        'pandas' for pandas DataFrame, 'json' for JSON format (default: 'pandas')
    start_date: str or datetime
        Start date for filtering data (format: 'YYYY-MM-DD' or datetime object)
    end_date: str or datetime
        End date for filtering data (format: 'YYYY-MM-DD' or datetime object)
        
    Returns:
    --------
    pandas DataFrame or dict: Monthly adjusted stock data
    """
    # API endpoint
    base_url = "https://www.alphavantage.co/query"
    
    # Request parameters
    params = {
        'function': 'TIME_SERIES_MONTHLY_ADJUSTED',
        'symbol': symbol,
        'outputsize': output_size,
        'apikey': ALPHA_API_KEY
    }
    
    print(f"Fetching monthly adjusted data for {symbol}...")
    
    # Make API request
    response = requests.get(base_url, params=params)
    
    # Check if request was successful
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
    
    # Parse response
    data = response.json()
    
    # Print the entire response for debugging
    print(f"API Response: {data}")
    
    # Check for rate limiting or API key issues
    if 'Note' in data:
        print(f"API Note: {data['Note']}")
        raise Exception("API rate limit reached. Please try again later or use a different API key.")
    
    # Check for error messages
    if 'Error Message' in data:
        raise Exception(f"API returned an error: {data['Error Message']}")
    
    # Check if data contains the expected time series
    time_series_key = None
    possible_keys = [
        'Monthly Adjusted Time Series',
        'Monthly Time Series',
        'Time Series (Monthly)',
        'Time Series (Monthly Adjusted)'
    ]
    
    for key in possible_keys:
        if key in data:
            time_series_key = key
            print(f"Found time series data under key: '{key}'")
            break
    
    if time_series_key is None:
        print(f"Available keys in response: {list(data.keys())}")
        raise Exception(f"No monthly time series found for {symbol}. Response: {data}")
    
    # Return as JSON if requested
    if output_format.lower() == 'json':
        return data
    
    # Convert to pandas DataFrame
    monthly_data = data[time_series_key]
    df = pd.DataFrame.from_dict(monthly_data, orient='index')
    
    # Convert string values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Rename columns for clarity
    df = df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. adjusted close': 'adj_close',
        '6. volume': 'volume',
        '7. dividend amount': 'dividend'
    })
    
    # Convert index to datetime and sort by date
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter by date range if provided
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    
    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    print(f"Successfully fetched {len(df)} months of data for {symbol}")
    
    return df

def save_to_csv(df, symbol, output_dir='stock_data'):
    """
    Save DataFrame to CSV file.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{symbol}_monthly_adjusted_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    
    return filepath

def plot_stock_data(df, symbol, output_dir='stock_data'):
    """
    Plot stock data and save the chart.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot adjusted close prices
    ax.plot(df.index, df['adj_close'], label='Adjusted Close')
    
    # Add title and labels
    ax.set_title(f'{symbol} Monthly Adjusted Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{symbol}_monthly_chart_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save chart
    plt.savefig(filepath)
    print(f"Chart saved to {filepath}")
    plt.close()
    
    return filepath

def analyze_monthly_returns(df, symbol):
    """
    Analyze monthly returns and provide summary statistics.
    """
    # Calculate monthly returns
    df['monthly_return'] = df['adj_close'].pct_change() * 100
    
    # Calculate summary statistics
    mean_return = df['monthly_return'].mean()
    median_return = df['monthly_return'].median()
    min_return = df['monthly_return'].min()
    max_return = df['monthly_return'].max()
    std_return = df['monthly_return'].std()
    
    print("\nMonthly Return Analysis:")
    print(f"Average Monthly Return: {mean_return:.2f}%")
    print(f"Median Monthly Return: {median_return:.2f}%")
    print(f"Minimum Monthly Return: {min_return:.2f}%")
    print(f"Maximum Monthly Return: {max_return:.2f}%")
    print(f"Standard Deviation: {std_return:.2f}%")
    
    # Calculate annualized return (geometric mean)
    monthly_return_factor = (1 + (df['monthly_return'] / 100))
    annualized_return = (monthly_return_factor.prod() ** (12 / len(df)) - 1) * 100
    
    print(f"Annualized Return: {annualized_return:.2f}%")
    
    return {
        'mean_return': mean_return,
        'median_return': median_return,
        'min_return': min_return,
        'max_return': max_return,
        'std_return': std_return,
        'annualized_return': annualized_return
    }

STOCK_LIST = ["RY.TO","INE.DB.B.TO",
              "SCR.TO", "NVA.TO", "CNE.TO", "VRY.TO", "TCW.TO", "WCP.TO", "IMO.TO", 
              "CJ.TO", "TPZ.TO", "BEP.PR.R.TO", "PSD.TO", "ENB.PR.D.TO", "ENB.PF.V.TO", 
              "ENB.PR.A.TO", "SGY.TO", "FRU.TO", "AQN.TO", "CEN.H.TO", "TRP.PR.F.TO", 
              "PPL.TO", "LTC.TO", "NSE.TO", "KEY.TO", "CFW.TO", "BNE.TO", "CVE.PR.A.TO", 
              "TVE.TO", "PAT.TO", "TRP.PR.E.TO", "TRP.PR.I.TO", "CVE.PR.G.TO", 
              "BEP.PR.M.TO", "PPL.PF.A.TO"]

def main(SYMBOL):
    # Configuration

    OUTPUT_DIR = "stock_data"
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Get stock data with date range
        df = get_monthly_adjusted_stock_data(SYMBOL, start_date=START_DATE, end_date=END_DATE)
        
        # Display the first few rows
        print("\nFirst 5 months of data:")
        print(df.head())
        
        # Save data to CSV
        csv_path = save_to_csv(df, SYMBOL, OUTPUT_DIR)
        
        # Plot stock data
        chart_path = plot_stock_data(df, SYMBOL, OUTPUT_DIR)
        
        # Analyze monthly returns
        return_stats = analyze_monthly_returns(df, SYMBOL)
        
        print(f"\nSuccessfully processed {SYMBOL} data from {START_DATE} to {END_DATE}.")
        print(f"Files saved to {OUTPUT_DIR} directory.")
        
    except Exception as e:
        print(f"Error: {e}")


import time 

for stock in STOCK_LIST:
    time.sleep(15)
    main(stock)

