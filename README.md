# Monte Carlo Stock Price Simulation 📈  

This project is a **Streamlit application** that uses the Monte Carlo method to simulate future stock prices and compares the results to the Black-Scholes option pricing model. It provides interactive visualizations and insights into stock price behaviors and option pricing.  

## Features  
- 📊 **Fetch Historical Stock Data**:  
  Use `yfinance` to retrieve closing prices of stocks over a selected date range.  

- 🎲 **Monte Carlo Simulation**:  
  Simulate future stock prices using Geometric Brownian Motion and visualize multiple simulated paths.  

- 🧮 **Black-Scholes Option Pricing**:  
  Calculate European call option prices and visualize their dependence on volatility and strike prices using a heatmap.  

- 📉 **Convergence Analysis**:  
  Track the convergence of simulated stock prices' cumulative average as the number of simulations increases.  

- 🛠️ **Interactive UI**:  
  - Adjust key parameters (e.g., volatility, risk-free rate, strike price) with easy-to-use sliders.  
  - Automatically fetch the latest stock price for a given ticker.  

## Technologies Used  
- **Python**  
- **Streamlit** for building the interactive user interface.  
- **yfinance** for fetching stock price data.  
- **NumPy** for numerical computations.  
- **Matplotlib** for plotting visualizations.  
- **SciPy** for statistical functions like cumulative distribution functions (CDFs).  


Access App using this link:
https://monte-carlo-options.streamlit.app
