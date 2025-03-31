# Investment Banking Risk-Return Analysis Application

This Streamlit application performs comprehensive risk-return analysis of investment banking stocks in India using historical data, CAPM model, and portfolio optimization techniques.

## Features

- **Stock Performance Analysis**: Visualize historical prices and returns of major Indian banking stocks
- **Risk-Return Metrics**: Calculate and visualize key risk and return metrics
- **CAPM Model**: Implement Capital Asset Pricing Model to analyze risk-adjusted returns
- **Portfolio Optimization**: Generate an efficient frontier and optimal portfolios
- **Correlation Analysis**: Analyze relationships between different banking stocks
- **Data Export**: Download all analysis results in Excel format

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/investment-banking-analysis.git
cd investment-banking-analysis
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the requirements:
```
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

## Application Structure

```
investment-banking-analysis/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## Data Sources

The application uses Yahoo Finance API through the `yfinance` library to retrieve historical stock data for Indian banking stocks and market indices.

## Analysis Methodology

1. **Data Collection**: Historical stock prices are retrieved from Yahoo Finance
2. **Returns Calculation**: Daily and monthly returns are computed
3. **Risk Metrics**: Standard deviation, variance, and other risk metrics are calculated
4. **CAPM Analysis**: Beta, alpha, and expected returns are computed using the CAPM model
5. **Portfolio Optimization**: Monte Carlo simulation is used to generate the efficient frontier
6. **Visualization**: Interactive charts and tables for data exploration

## References

This application is based on financial theories and methodologies discussed in:

- "Risk and Return Analysis of Selected Banking Stocks in India" (2019)
- Modern Portfolio Theory by Harry Markowitz
- Capital Asset Pricing Model (CAPM) by William Sharpe