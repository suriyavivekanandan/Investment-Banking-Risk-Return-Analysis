import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Set whether to show debug information
DEBUG_MODE = True

# Set page config
st.set_page_config(
    page_title="Investment Banking Risk-Return Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem 1rem;
    }
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 28px;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .caption {
        font-size: 14px;
        color: #424242;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# Define the list of Indian bank stocks to analyze
BANK_STOCKS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS", 
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Yes Bank": "YESBANK.NS",
    "Federal Bank": "FEDERALBNK.NS",
    "Punjab National Bank": "PNB.NS",
    "Bank of Baroda": "BANKBARODA.NS"
}

# Market index for CAPM
MARKET_INDEX = "^NSEI"  # Nifty 50 Index

def check_internet_connection():
    """Check if internet connection is available"""
    try:
        # Try connecting to a reliable server
        requests.get("https://www.google.com", timeout=5)
        return True
    except:
        return False

def fetch_data(tickers, start_date, end_date, max_retries=3):
    """Fetch stock data for given tickers and date range with retry logic"""
    if not check_internet_connection():
        st.error("No internet connection detected. Please check your network and try again.")
        return None
        
    if DEBUG_MODE:
        st.write(f"Attempting to fetch data for tickers: {tickers}")
        st.write(f"Date range: {start_date} to {end_date}")
    
    for attempt in range(max_retries):
        try:
            # Create a session with custom User-Agent
            session = requests.Session()
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            session.headers['User-Agent'] = user_agent
            
            # Download data using the session parameter instead of headers
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                progress=False,
                session=session  # Use session parameter instead of headers
            )
            
            # Check if data is empty
            if data.empty:
                if attempt < max_retries - 1:
                    if DEBUG_MODE:
                        st.warning(f"No data found. Retrying... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    st.error("No data found for the selected stocks and date range.")
                    return None
            
            # Handle multindex vs single index case
            if isinstance(data.columns, pd.MultiIndex):
                # Check if 'Adj Close' exists in the data
                if 'Adj Close' in data.columns.levels[0]:
                    # Extract Adj Close data
                    adj_close_data = data['Adj Close']
                else:
                    # If 'Adj Close' doesn't exist, silently use 'Close' instead
                    if 'Close' in data.columns.levels[0]:
                        adj_close_data = data['Close']
                    else:
                        st.error("Price data not available for the selected tickers.")
                        return None
            else:
                # For single ticker, data won't have MultiIndex columns
                if 'Adj Close' in data.columns:
                    adj_close_data = data['Adj Close'].to_frame(name=tickers[0])
                else:
                    # Try using 'Close' if 'Adj Close' is not available
                    if 'Close' in data.columns:
                        adj_close_data = data['Close'].to_frame(name=tickers[0])
                    else:
                        st.error("Price data not available for the selected ticker.")
                        return None
            
            # Check if we have actual data in the dataframe
            if adj_close_data.empty or adj_close_data.isnull().all().all():
                if attempt < max_retries - 1:
                    if DEBUG_MODE:
                        st.warning(f"Retrieved data contains only NaN values. Retrying... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    st.error("Data retrieved contains only NaN values. Please check tickers or date range.")
                    return None
                
            if DEBUG_MODE:
                st.success(f"Successfully fetched data with shape: {adj_close_data.shape}")
            
            return adj_close_data
            
        except Exception as e:
            if attempt < max_retries - 1:
                if DEBUG_MODE:
                    st.warning(f"Error fetching data: {str(e)}. Retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(2)  # Wait before retrying
            else:
                st.error(f"Error fetching data: {str(e)}")
                return None
                
    return None

def calculate_returns(prices_df):
    """Calculate daily and monthly returns from price data"""
    # Daily returns
    daily_returns = prices_df.pct_change().dropna()
    
    # Monthly returns (resample to month end and calculate percentage change)
    monthly_returns = prices_df.resample('M').last().pct_change().dropna()
    
    return daily_returns, monthly_returns

def calculate_risk_metrics(returns_df):
    """Calculate risk metrics including standard deviation, variance, etc."""
    # Annualized metrics (assuming 252 trading days, 12 months)
    risk_metrics = pd.DataFrame({
        'Mean Daily Return (%)': returns_df.mean() * 100,
        'Daily Risk (Std Dev) (%)': returns_df.std() * 100,
        'Annualized Return (%)': (returns_df.mean() * 252) * 100,
        'Annualized Risk (%)': (returns_df.std() * np.sqrt(252)) * 100,
        'Variance': returns_df.var(),
    })
    
    return risk_metrics

def calculate_capm(stock_returns, market_returns, risk_free_rate):
    """Calculate CAPM metrics including beta, alpha, expected return"""
    results = {}
    
    # For each stock in our dataset
    for column in stock_returns.columns:
        # Calculate Beta using covariance method
        covariance = stock_returns[column].cov(market_returns)
        market_variance = market_returns.var()
        beta = covariance / market_variance
        
        # Expected return according to CAPM
        expected_return = risk_free_rate + beta * (market_returns.mean() * 252 - risk_free_rate)
        
        # Calculate Alpha (Jensen's Alpha)
        actual_return = stock_returns[column].mean() * 252
        alpha = actual_return - expected_return
        
        # Calculate R-squared (how much of the stock's movement is explained by the market)
        correlation = stock_returns[column].corr(market_returns)
        r_squared = correlation ** 2
        
        # Store results
        results[column] = {
            'Beta': beta,
            'Expected Return (%)': expected_return * 100,
            'Actual Return (%)': actual_return * 100,
            'Alpha (%)': alpha * 100,
            'R-squared': r_squared
        }
    
    return pd.DataFrame(results).T

def create_efficient_frontier(returns, num_portfolios=1000):
    """Generate random portfolios for efficient frontier visualization"""
    # Number of assets
    n_assets = len(returns.columns)
    
    # Arrays to store results
    port_returns = np.zeros(num_portfolios)
    port_volatilities = np.zeros(num_portfolios)
    sharpe_ratios = np.zeros(num_portfolios)
    stock_weights = np.zeros((num_portfolios, n_assets))
    
    # Assuming risk-free rate of 3.5%
    risk_free_rate = 0.035
    
    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean() * 252  # Annualized
    cov_matrix = returns.cov() * 252     # Annualized
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        stock_weights[i, :] = weights
        
        # Calculate portfolio return and volatility
        port_returns[i] = np.sum(mean_returns * weights)
        port_volatilities[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratios[i] = (port_returns[i] - risk_free_rate) / port_volatilities[i]
    
    # Create a DataFrame with the results
    portfolio_results = pd.DataFrame({
        'Return': port_returns,
        'Volatility': port_volatilities,
        'Sharpe Ratio': sharpe_ratios
    })
    
    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_idx = portfolio_results['Sharpe Ratio'].idxmax()
    max_sharpe_port = {
        'Return': portfolio_results.loc[max_sharpe_idx, 'Return'],
        'Volatility': portfolio_results.loc[max_sharpe_idx, 'Volatility'],
        'Sharpe Ratio': portfolio_results.loc[max_sharpe_idx, 'Sharpe Ratio'],
        'Weights': stock_weights[max_sharpe_idx, :]
    }
    
    # Find the portfolio with the minimum volatility
    min_vol_idx = portfolio_results['Volatility'].idxmin()
    min_vol_port = {
        'Return': portfolio_results.loc[min_vol_idx, 'Return'],
        'Volatility': portfolio_results.loc[min_vol_idx, 'Volatility'],
        'Sharpe Ratio': portfolio_results.loc[min_vol_idx, 'Sharpe Ratio'],
        'Weights': stock_weights[min_vol_idx, :]
    }
    
    return portfolio_results, max_sharpe_port, min_vol_port, returns.columns

def plot_stock_prices(prices_df):
    """Plot stock prices over time"""
    fig = px.line(prices_df, x=prices_df.index, y=prices_df.columns,
                  title="Stock Prices Over Time",
                  labels={"value": "Price (INR)", "variable": "Stock"})
    fig.update_layout(legend_title_text='Banks', height=500)
    return fig

def plot_returns(returns_df, period="Daily"):
    """Plot returns over time"""
    fig = px.line(returns_df, x=returns_df.index, y=returns_df.columns,
                  title=f"{period} Returns Over Time",
                  labels={"value": f"{period} Return (%)", "variable": "Stock"})
    fig.update_layout(legend_title_text='Banks', height=500)
    return fig

def plot_risk_return_scatter(risk_metrics):
    """Create a risk-return scatter plot"""
    # Create a copy of the data to avoid modifying the original
    plot_data = risk_metrics.copy()
    
    # Create a size column that's guaranteed to be positive
    # Either use absolute value or ensure a minimum positive value
    plot_data['Size'] = plot_data['Annualized Return (%)'].apply(lambda x: max(abs(x), 5))
    
    fig = px.scatter(
        plot_data, 
        x="Annualized Risk (%)", 
        y="Annualized Return (%)",
        text=plot_data.index,
        size="Size",  # Use the new size column
        color="Annualized Return (%)",
        title="Risk-Return Profile of Indian Banking Stocks",
        height=600,
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title="Risk (Annual Standard Deviation %)", 
                      yaxis_title="Return (Annual %)")
    return fig

def plot_efficient_frontier(portfolio_results, max_sharpe_port, min_vol_port, stock_risk_return):
    """Plot the efficient frontier with individual stocks"""
    fig = go.Figure()
    
    # Add scatter plot for thousands of portfolios
    fig.add_trace(go.Scatter(
        x=portfolio_results['Volatility'], 
        y=portfolio_results['Return'],
        mode='markers',
        marker=dict(
            size=5,
            color=portfolio_results['Sharpe Ratio'],
            colorscale='Viridis',
            colorbar=dict(title="Sharpe Ratio"),
            showscale=True
        ),
        name='Portfolios'
    ))
    
    # Add individual stocks
    # Convert percentage values back to decimals for consistency with portfolio_results
    x_vals = stock_risk_return['Annualized Risk (%)'] / 100
    y_vals = stock_risk_return['Annualized Return (%)'] / 100
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=stock_risk_return.index,
        textposition="top center",
        name='Individual Stocks'
    ))
    
    # Add the max Sharpe ratio portfolio
    fig.add_trace(go.Scatter(
        x=[max_sharpe_port['Volatility']], 
        y=[max_sharpe_port['Return']],
        mode='markers',
        marker=dict(size=15, color='green', symbol='star'),
        name='Max Sharpe Ratio'
    ))
    
    # Add the min volatility portfolio
    fig.add_trace(go.Scatter(
        x=[min_vol_port['Volatility']], 
        y=[min_vol_port['Return']],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='star'),
        name='Min Volatility'
    ))
    
    fig.update_layout(
        title='Efficient Frontier with Individual Banking Stocks',
        xaxis_title='Annualized Volatility (Standard Deviation)',
        yaxis_title='Annualized Return',
        height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def plot_correlation_heatmap(returns_df):
    """Plot correlation heatmap of stock returns"""
    corr_matrix = returns_df.corr()
    fig = px.imshow(corr_matrix, 
                    x=corr_matrix.columns, 
                    y=corr_matrix.columns, 
                    color_continuous_scale="Blues",
                    title="Correlation Matrix of Stock Returns")
    fig.update_layout(height=600)
    return fig

def plot_capm_chart(capm_results):
    """Plot CAPM results"""
    # Create a figure with two subplots
    fig = go.Figure()
    
    # Add Beta vs Expected Return trace
    fig.add_trace(go.Scatter(
        x=capm_results['Beta'],
        y=capm_results['Expected Return (%)'],
        mode='markers+text',
        marker=dict(size=12, color='blue'),
        text=capm_results.index,
        textposition="top center",
        name='Beta vs Expected Return'
    ))
    
    # Add a trendline
    z = np.polyfit(capm_results['Beta'], capm_results['Expected Return (%)'], 1)
    y_hat = np.poly1d(z)(capm_results['Beta'])
    
    fig.add_trace(go.Scatter(
        x=capm_results['Beta'],
        y=y_hat,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Security Market Line'
    ))
    
    fig.update_layout(
        title='CAPM: Beta vs Expected Return',
        xaxis_title='Beta',
        yaxis_title='Expected Return (%)',
        height=600
    )
    
    return fig

def main():
    st.markdown('<div class="title">Investment Banking Risk-Return Analysis</div>', unsafe_allow_html=True)
    
    st.write("""
    This application performs a comprehensive risk-return analysis of major Indian banking stocks using 
    historical data. The analysis includes CAPM models, efficient frontier optimization, and various financial metrics.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Parameters")
    
    # Date range selection - modified for more reliable date ranges
    today = datetime.today()
    default_start_date = today - timedelta(days=365*2)  # 2 years ago by default (reduced from 3)
    max_end_date = today - timedelta(days=1)  # Yesterday instead of today
    
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", max_end_date)
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
        return
    
    # Risk-free rate input
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=1.0, max_value=10.0, value=3.5) / 100
    
    # Bank selection - reduced default selection
    selected_banks = st.sidebar.multiselect(
        "Select Banks to Analyze",
        options=list(BANK_STOCKS.keys()),
        default=list(BANK_STOCKS.keys())[:3]  # Default to first 3 banks instead of 5
    )
    
    if not selected_banks:
        st.warning("Please select at least one bank to analyze.")
        return
    
    # Get the tickers for selected banks
    selected_tickers = [BANK_STOCKS[bank] for bank in selected_banks]
    
    # Add market index
    all_tickers = selected_tickers + [MARKET_INDEX]
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching stock data...")
    # Get stock price data with improved error handling
    prices_df = fetch_data(all_tickers, start_date, end_date)
    progress_bar.progress(20)
    
    if prices_df is None or prices_df.empty:
        st.error("Failed to fetch data. Please check your inputs and try again.")
        
        # Provide troubleshooting suggestions
        st.markdown("""
        ### Troubleshooting Suggestions:
        1. Try selecting fewer stocks at once
        2. Use a shorter date range (1 year instead of multiple years)
        3. Avoid very recent dates - data might not be available yet
        4. Check your internet connection
        5. Try again in a few minutes - Yahoo Finance API might be temporarily unavailable
        """)
        return
    
    # Separate market data
    market_prices = prices_df[MARKET_INDEX]
    stock_prices = prices_df.drop(columns=[MARKET_INDEX])
    
    status_text.text("Calculating returns...")
    # Calculate returns
    daily_returns, monthly_returns = calculate_returns(stock_prices)
    market_daily_returns = market_prices.pct_change().dropna()
    progress_bar.progress(40)
    
    status_text.text("Calculating risk metrics...")
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(daily_returns)
    progress_bar.progress(60)
    
    status_text.text("Running CAPM analysis...")
    # Calculate CAPM metrics
    capm_results = calculate_capm(daily_returns, market_daily_returns, risk_free_rate)
    progress_bar.progress(80)
    
    status_text.text("Generating efficient frontier...")
    # Calculate efficient frontier
    portfolio_results, max_sharpe_port, min_vol_port, portfolio_assets = create_efficient_frontier(daily_returns)
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Stock Performance", 
        "📊 Risk-Return Analysis", 
        "🧮 CAPM Model", 
        "🎯 Portfolio Optimization",
        "📉 Correlations",
        "📑 Data Download"
    ])
    
    with tab1:
        st.markdown('<div class="subtitle">Stock Price Performance</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_stock_prices(stock_prices), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="subtitle">Daily Returns</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_returns(daily_returns, "Daily"), use_container_width=True)
        
        with col2:
            st.markdown('<div class="subtitle">Monthly Returns</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_returns(monthly_returns, "Monthly"), use_container_width=True)
    
    with tab2:
        st.markdown('<div class="subtitle">Risk-Return Metrics</div>', unsafe_allow_html=True)
        st.dataframe(risk_metrics.style.format({
            'Mean Daily Return (%)': '{:.4f}',
            'Daily Risk (Std Dev) (%)': '{:.4f}',
            'Annualized Return (%)': '{:.2f}',
            'Annualized Risk (%)': '{:.2f}',
            'Variance': '{:.6f}'
        }))
        
        st.markdown('<div class="subtitle">Risk-Return Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_risk_return_scatter(risk_metrics), use_container_width=True)
    
    with tab3:
        st.markdown('<div class="subtitle">Capital Asset Pricing Model (CAPM) Results</div>', unsafe_allow_html=True)
        st.dataframe(capm_results.style.format({
            'Beta': '{:.4f}',
            'Expected Return (%)': '{:.2f}',
            'Actual Return (%)': '{:.2f}',
            'Alpha (%)': '{:.2f}',
            'R-squared': '{:.4f}'
        }))
        
        st.markdown('<div class="caption">Beta > 1: Stock is more volatile than the market</div>', unsafe_allow_html=True)
        st.markdown('<div class="caption">Beta < 1: Stock is less volatile than the market</div>', unsafe_allow_html=True)
        st.markdown('<div class="caption">Alpha > 0: Stock performed better than expected given its risk</div>', unsafe_allow_html=True)
        st.markdown('<div class="caption">Alpha < 0: Stock performed worse than expected given its risk</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subtitle">CAPM Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_capm_chart(capm_results), use_container_width=True)
    
    with tab4:
        st.markdown('<div class="subtitle">Portfolio Optimization</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_efficient_frontier(portfolio_results, max_sharpe_port, min_vol_port, risk_metrics), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="subtitle">Optimal Portfolio (Maximum Sharpe Ratio)</div>', unsafe_allow_html=True)
            st.write(f"Expected Annual Return: {max_sharpe_port['Return']*100:.2f}%")
            st.write(f"Expected Annual Volatility: {max_sharpe_port['Volatility']*100:.2f}%")
            st.write(f"Sharpe Ratio: {max_sharpe_port['Sharpe Ratio']:.4f}")
            
            # Create a pie chart for portfolio weights
            weights_df = pd.DataFrame({
                'Stock': portfolio_assets,
                'Weight': max_sharpe_port['Weights'] * 100
            })
            fig = px.pie(weights_df, values='Weight', names='Stock', title='Optimal Portfolio Weights')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="subtitle">Minimum Volatility Portfolio</div>', unsafe_allow_html=True)
            st.write(f"Expected Annual Return: {min_vol_port['Return']*100:.2f}%")
            st.write(f"Expected Annual Volatility: {min_vol_port['Volatility']*100:.2f}%")
            st.write(f"Sharpe Ratio: {min_vol_port['Sharpe Ratio']:.4f}")
            
            # Create a pie chart for portfolio weights
            weights_df = pd.DataFrame({
                'Stock': portfolio_assets,
                'Weight': min_vol_port['Weights'] * 100
            })
            fig = px.pie(weights_df, values='Weight', names='Stock', title='Minimum Volatility Portfolio Weights')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<div class="subtitle">Correlation Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_correlation_heatmap(daily_returns), use_container_width=True)
        
        st.markdown("""
        <div class="caption">
        Higher correlation between stocks (darker blue) indicates less diversification benefit when these stocks are combined in a portfolio.
        </div>
        """, unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<div class="subtitle">Download Analysis Data</div>', unsafe_allow_html=True)
        
        # Create Excel file with multiple sheets
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Stock prices
            stock_prices.to_excel(writer, sheet_name='Stock Prices')
            
            # Returns
            daily_returns.to_excel(writer, sheet_name='Daily Returns')
            monthly_returns.to_excel(writer, sheet_name='Monthly Returns')
            
            # Risk metrics
            risk_metrics.to_excel(writer, sheet_name='Risk Metrics')
            
            # CAPM results
            capm_results.to_excel(writer, sheet_name='CAPM Results')
            
            # Correlation matrix
            daily_returns.corr().to_excel(writer, sheet_name='Correlation Matrix')
            
            # Portfolio optimization
            # Optimal portfolio (max Sharpe ratio)
            max_sharpe_weights = pd.DataFrame({
                'Stock': portfolio_assets,
                'Weight (%)': max_sharpe_port['Weights'] * 100
            })
            max_sharpe_weights.to_excel(writer, sheet_name='Optimal Portfolio')
            
            # Min volatility portfolio
            min_vol_weights = pd.DataFrame({
                'Stock': portfolio_assets,
                'Weight (%)': min_vol_port['Weights'] * 100
            })
            min_vol_weights.to_excel(writer, sheet_name='Min Vol Portfolio')
        
        output.seek(0)
        
        st.download_button(
            label="Download Excel Report",
            data=output,
            file_name=f"banking_risk_return_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.markdown("""
        <div class="caption">
        The Excel report contains all the analyzed data including stock prices, returns, risk metrics, CAPM results, correlation matrix, and optimized portfolio weights.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()