import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time
import requests

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

# Set up a custom session with headers to avoid Yahoo Finance blocking
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    })
    return session

@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce API calls
def fetch_data(tickers, start_date, end_date, max_retries=3):
    """Fetch stock data for given tickers and date range with retries"""
    session = get_yf_session()
    
    for attempt in range(max_retries):
        try:
            st.info(f"Fetching data (attempt {attempt+1}/{max_retries})...")
            
            # Download all data with custom session and increased timeout
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date,
                progress=False,
                session=session,
                timeout=30
            )
            
            # Check if data is empty
            if data.empty:
                st.warning(f"No data found on attempt {attempt+1}. Retrying...")
                time.sleep(2 * (attempt + 1))  # Exponential backoff
                continue
                
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
                        st.warning(f"Price data not available on attempt {attempt+1}. Retrying...")
                        time.sleep(2 * (attempt + 1))
                        continue
            else:
                # For single ticker, data won't have MultiIndex columns
                if 'Adj Close' in data.columns:
                    adj_close_data = data['Adj Close'].to_frame(name=tickers[0])
                else:
                    # Try using 'Close' if 'Adj Close' is not available
                    if 'Close' in data.columns:
                        adj_close_data = data['Close'].to_frame(name=tickers[0])
                    else:
                        st.warning(f"Price data not available on attempt {attempt+1}. Retrying...")
                        time.sleep(2 * (attempt + 1))
                        continue
            
            # Check if we have actual data in the dataframe
            if adj_close_data.empty or adj_close_data.isnull().all().all():
                st.warning(f"Retrieved data contains only NaN values on attempt {attempt+1}. Retrying...")
                time.sleep(2 * (attempt + 1))
                continue
                
            st.success("Data successfully fetched!")
            return adj_close_data
            
        except Exception as e:
            st.warning(f"Error on attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                st.error(f"Failed to fetch data after {max_retries} attempts. Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())  # Print stack trace
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
    
    # Add troubleshooting info
    with st.expander("⚠️ Troubleshooting Data Issues"):
        st.write("""
        - If data doesn't load, try selecting a smaller date range or fewer banks.
        - The app needs to connect to Yahoo Finance API, which might occasionally have connectivity issues.
        - First-time loading can take longer as data is being fetched and cached.
        """)
    
    # Sidebar for inputs
    st.sidebar.header("Analysis Parameters")
    
    # Date range selection
    today = datetime.today()
    default_start_date = today - timedelta(days=365*3)  # 3 years ago by default
    
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", today)
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
        return
    
    # Risk-free rate input
    risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=1.0, max_value=10.0, value=3.5) / 100
    
    # Bank selection
    selected_banks = st.sidebar.multiselect(
        "Select Banks to Analyze",
        options=list(BANK_STOCKS.keys()),
        default=list(BANK_STOCKS.keys())[:5]  # Default to first 5 banks
    )
    
    if not selected_banks:
        st.warning("Please select at least one bank to analyze.")
        return
    
    # Get the tickers for selected banks
    selected_tickers = [BANK_STOCKS[bank] for bank in selected_banks]
    
    # Add market index
    all_tickers = selected_tickers + [MARKET_INDEX]
    
    # Check connectivity to Yahoo Finance before proceeding
    st.info("Testing connection to Yahoo Finance...")
    try:
        test_data = yf.download(MARKET_INDEX, period="1d", progress=False)
        if test_data.empty:
            st.warning("Connection test returned empty data. Will still attempt to fetch actual data.")
        else:
            st.success("Connection to Yahoo Finance successful! Proceeding with analysis.")
    except Exception as e:
        st.warning(f"Connection test warning: {str(e)}. Will still attempt to fetch actual data.")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Fetching stock data...")
    # Get stock price data
    prices_df = fetch_data(all_tickers, start_date, end_date)
    progress_bar.progress(20)
    
    if prices_df is None or prices_df.empty:
        st.error("Failed to fetch data. Please check your inputs and try again later.")
        
        # Add fallback option
        st.warning("Would you like to try with sample data instead?")
        if st.button("Use Sample Data"):
            # Generate sample data
            sample_dates = pd.date_range(start=start_date, end=end_date, freq='B')
            sample_data = {}
            
            np.random.seed(42)  # For reproducibility
            for ticker in all_tickers:
                # Generate random walk with drift
                returns = np.random.normal(0.0005, 0.015, size=len(sample_dates))
                price = 100 * (1 + returns).cumprod()
                sample_data[ticker] = price
            
            prices_df = pd.DataFrame(sample_data, index=sample_dates)
            st.success("Sample data generated successfully!")
        else:
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

    # Add footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>Note: This application uses Yahoo Finance data which may occasionally face connectivity issues when deployed on cloud platforms.</p>
    <p>If you encounter persistent data fetching issues, please try again later or with a smaller date range.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()