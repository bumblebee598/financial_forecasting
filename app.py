import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants
DATA_DIR = 'stock_data'  # Directory containing CSV files
MODEL_RESULTS_DIR = 'model_results'  # Directory with model results
MODELS_DIR = 'models'  # Directory for saved models

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.3rem;
        color: #1565C0;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
    }
    .info-box {
        background-color: rgba(144, 238, 144, 0.2);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .prediction-up {
        color: green;
        font-weight: bold;
    }
    .prediction-down {
        color: red;
        font-weight: bold;
    }
    .table-container {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_available_stocks():
    """
    Load the list of available stocks from CSV files
    """
    pattern = f"{DATA_DIR}/*_monthly_adjusted_*.csv"
    stock_files = glob.glob(pattern)
    
    stocks = []
    for file_path in stock_files:
        file_name = os.path.basename(file_path)
        symbol = file_name.split('_')[0]
        if symbol not in stocks:
            stocks.append(symbol)
    
    return sorted(stocks)

@st.cache_data
def load_stock_data(symbol):
    """
    Load stock data for a specific symbol
    """
    pattern = f"{DATA_DIR}/{symbol}_monthly_adjusted_*.csv"
    latest_file = max(glob.glob(pattern), key=os.path.getctime, default=None)
    
    if not latest_file:
        st.error(f"No data found for {symbol}")
        return None
    
    # Load CSV
    df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
    
    # Sort by date (ascending)
    df = df.sort_index()
    
    return df

@st.cache_data
def load_model_metrics(symbol):
    """
    Load model comparison metrics for a specific symbol
    """
    metrics_file = f"{MODEL_RESULTS_DIR}/{symbol}_model_comparison.csv"
    
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    else:
        return None

@st.cache_data
def load_model_predictions(symbol):
    """
    Load model predictions for plotting
    """
    pred_pickle = f"{MODEL_RESULTS_DIR}/{symbol}_predictions.pkl"
    
    if os.path.exists(pred_pickle):
        with open(pred_pickle, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def format_number(num, prefix="$"):
    """
    Format numbers for display (e.g., $1.23K, $1.23M)
    """
    if num is None:
        return "N/A"
    
    if abs(num) >= 1_000_000_000:
        return f"{prefix}{num/1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"{prefix}{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000:
        return f"{prefix}{num/1_000:.2f}K"
    else:
        return f"{prefix}{num:.2f}"

def calculate_returns(data, periods):
    """
    Calculate returns over different periods
    """
    returns = {}
    
    if data is None or len(data) < max(periods):
        return {p: None for p in periods}
    
    latest_price = data['adj_close'].iloc[-1]
    
    for period in periods:
        if period <= len(data):
            past_price = data['adj_close'].iloc[-period]
            returns[period] = (latest_price / past_price - 1) * 100
        else:
            returns[period] = None
    
    return returns

def create_stock_price_chart(data, symbol):
    """
    Create an interactive stock price chart with volume
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['adj_close'],
            name='Adjusted Close',
            line=dict(color='#1E88E5', width=2)
        ),
        secondary_y=False
    )
    
    # Add volume bars if available
    if 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                opacity=0.3,
                marker=dict(color='#0D47A1')
            ),
            secondary_y=True
        )
    
    # Customize layout
    fig.update_layout(
        title=f"{symbol} Stock Price History",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price", secondary_y=False)
    if 'volume' in data.columns:
        fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    return fig

def create_returns_chart(data, symbol):
    """
    Create a chart showing monthly returns
    """
    if data is None or len(data) < 2:
        return None
    
    # Calculate monthly returns
    monthly_returns = data['adj_close'].pct_change() * 100
    
    # Create a DataFrame with year and month
    returns_df = pd.DataFrame({
        'year': data.index.year,
        'month': data.index.month,
        'return': monthly_returns
    }).dropna()
    
    # Create pivot table for heatmap
    pivot_returns = returns_df.pivot_table(
        index='year', 
        columns='month', 
        values='return'
    )
    
    # Create heatmap using plotly
    fig = px.imshow(
        pivot_returns,
        labels=dict(x="Month", y="Year", color="Return (%)"),
        x=[f"{m}" for m in range(1, 13)],
        y=pivot_returns.index,
        color_continuous_scale="RdBu_r",
        origin='lower',
        aspect="auto"
    )
    
    fig.update_layout(
        title=f"{symbol} Monthly Returns (%)",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_model_comparison_chart(metrics):
    """
    Create a bar chart comparing model performance
    """
    if metrics is None:
        return None
    
    # Create bar chart using plotly
    fig = go.Figure()
    
    # Add bars for each metric
    metrics_to_plot = ['RMSE', 'MAE']
    colors = ['#1E88E5', '#FFC107']
    
    for i, metric in enumerate(metrics_to_plot):
        fig.add_trace(go.Bar(
            x=metrics['Model'],
            y=metrics[metric],
            name=metric,
            marker_color=colors[i]
        ))
    
    # Customize layout
    fig.update_layout(
        title="Model Performance Comparison (Lower is Better)",
        xaxis_title="Model",
        yaxis_title="Error Metric",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_prediction_chart(data, predictions, symbol):
    """
    Create chart with historical data and predictions from all models
    """
    if data is None or predictions is None:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['adj_close'],
        name='Historical',
        line=dict(color='black', width=2)
    ))
    
    # Add vertical line for current date
    last_date = data.index[-1]
    fig.add_vline(
        x=last_date, 
        line_width=2, 
        line_dash="dash", 
        line_color="gray"
    )
    
    # Add prediction for each model
    colors = {
        'XGBoost': 'blue',
        'LSTM': 'green',
        'Prophet': 'red'
    }
    
    for model_name, pred_data in predictions.items():
        if 'dates' in pred_data and 'values' in pred_data:
            fig.add_trace(go.Scatter(
                x=pred_data['dates'],
                y=pred_data['values'],
                name=f"{model_name} Prediction",
                line=dict(color=colors.get(model_name, 'orange'), dash='dash')
            ))
    
    # Customize layout
    fig.update_layout(
        title=f"{symbol} Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add annotation for prediction start
    fig.add_annotation(
        x=last_date,
        y=data['adj_close'].iloc[-1],
        text="Prediction Start",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    return fig

def main():
    """
    Main function to run the dashboard
    """
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for stock selection
    st.sidebar.title("Stock Selection")
    
    # Load available stocks
    available_stocks = load_available_stocks()
    
    if not available_stocks:
        st.error("No stock data found. Please run the data collection script first.")
        return
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select a stock:",
        available_stocks,
        index=0
    )
    
    # Stock search box
    stock_search = st.sidebar.text_input("Or search for a stock symbol:")
    if stock_search and stock_search.upper() in available_stocks:
        selected_stock = stock_search.upper()
    
    # Load data for selected stock
    stock_data = load_stock_data(selected_stock)
    
    if stock_data is None:
        st.error(f"No data found for {selected_stock}")
        return
    
    # Load model metrics
    model_metrics = load_model_metrics(selected_stock)
    
    # Load model predictions
    model_predictions = load_model_predictions(selected_stock)
    
    # Sidebar - Date Range Selection
    st.sidebar.title("Date Range")
    
    min_date = stock_data.index.min().date()
    max_date = stock_data.index.max().date()
    
    start_date = st.sidebar.date_input(
        "Start date",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End date",
        max_date,
        min_value=start_date,
        max_value=max_date
    )
    
    # Filter data by date range
    filtered_data = stock_data.loc[start_date:end_date].copy()
    
    # Calculate returns
    returns = calculate_returns(stock_data, [1, 3, 6, 12, 36, 60])
    
    # Dashboard Layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{format_number(stock_data["adj_close"].iloc[-1])}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Latest Price</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        month_return = returns.get(1)
        color = "prediction-up" if month_return and month_return > 0 else "prediction-down"
        st.markdown(f'<div class="metric-value {color}">{month_return:.2f}%</div>' if month_return is not None else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">1-Month Return</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        year_return = returns.get(12)
        color = "prediction-up" if year_return and year_return > 0 else "prediction-down"
        st.markdown(f'<div class="metric-value {color}">{year_return:.2f}%</div>' if year_return is not None else '<div class="metric-value">N/A</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">1-Year Return</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        if model_predictions and any('values' in pred for pred in model_predictions.values()):
            # Get the prediction from the best model (lowest RMSE)
            best_model = model_metrics.iloc[0]['Model'] if model_metrics is not None else list(model_predictions.keys())[0]
            
            if best_model in model_predictions and 'values' in model_predictions[best_model]:
                future_price = model_predictions[best_model]['values'][-1]
                current_price = stock_data['adj_close'].iloc[-1]
                pred_return = ((future_price / current_price) - 1) * 100
                
                color = "prediction-up" if pred_return > 0 else "prediction-down"
                st.markdown(f'<div class="metric-value {color}">{pred_return:.2f}%</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Predicted Return ({best_model})</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Predicted Return</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">N/A</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predicted Return</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Price Chart
    st.markdown('<h2 class="sub-header">Stock Price History</h2>', unsafe_allow_html=True)
    price_chart = create_stock_price_chart(filtered_data, selected_stock)
    st.plotly_chart(price_chart, use_container_width=True)
    
    # Tabs for additional content
    tab1, tab2, tab3 = st.tabs(["📊 Stock Data", "🔮 Predictions", "📈 Returns Analysis"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Stock Data Table</h2>', unsafe_allow_html=True)
        
        # Format data for display
        display_data = filtered_data.copy()
        display_data.index = display_data.index.strftime('%Y-%m-%d')
        
        # Sort by date (newest first for display)
        display_data = display_data.sort_index(ascending=False)
        
        # Display the table
        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        st.dataframe(display_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button for data
        csv = filtered_data.to_csv()
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{selected_stock}_data.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown('<h2 class="section-header">Model Predictions</h2>', unsafe_allow_html=True)
        
        if model_metrics is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown("### Model Performance")
                st.dataframe(model_metrics, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.write("### Best Model:")
                best_model = model_metrics.iloc[0]['Model']
                st.write(f"**{best_model}** with RMSE: {model_metrics.iloc[0]['RMSE']:.4f}")
                
                # Additional model info
                st.write("### Model Information:")
                if best_model == "XGBoost":
                    st.write("- Tree-based gradient boosting algorithm")
                    st.write("- Captures non-linear relationships")
                    st.write("- Good with tabular financial data")
                elif best_model == "LSTM":
                    st.write("- Long Short-Term Memory neural network")
                    st.write("- Captures time series patterns")
                    st.write("- Good for long-term dependencies")
                elif best_model == "Prophet":
                    st.write("- Decomposable time series model")
                    st.write("- Captures seasonality trends")
                    st.write("- Handles missing data well")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Model comparison chart
                model_chart = create_model_comparison_chart(model_metrics)
                if model_chart:
                    st.plotly_chart(model_chart, use_container_width=True)
                
                # Prediction chart
                if model_predictions:
                    pred_chart = create_prediction_chart(stock_data, model_predictions, selected_stock)
                    if pred_chart:
                        st.plotly_chart(pred_chart, use_container_width=True)
        else:
            st.info("No model predictions available for this stock. Please run the prediction model first.")
    
    with tab3:
        st.markdown('<h2 class="section-header">Returns Analysis</h2>', unsafe_allow_html=True)
        
        # Returns heatmap
        returns_chart = create_returns_chart(stock_data, selected_stock)
        if returns_chart:
            st.plotly_chart(returns_chart, use_container_width=True)
        
        # Period returns
        st.markdown("### Returns by Period")
        periods = [
            {"name": "1 Month", "months": 1},
            {"name": "3 Months", "months": 3},
            {"name": "6 Months", "months": 6},
            {"name": "1 Year", "months": 12},
            {"name": "3 Years", "months": 36},
            {"name": "5 Years", "months": 60}
        ]
        
        col1, col2, col3 = st.columns(3)
        
        for i, period in enumerate(periods):
            with [col1, col2, col3][i % 3]:
                ret = returns.get(period["months"])
                if ret is not None:
                    color = "green" if ret > 0 else "red"
                    st.metric(
                        label=period["name"], 
                        value=f"{ret:.2f}%",
                        delta=None
                    )
                else:
                    st.metric(
                        label=period["name"], 
                        value="N/A",
                        delta=None
                    )
    
    # Footer
    st.markdown('<div class="footer">Stock Prediction Dashboard | Created with Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
