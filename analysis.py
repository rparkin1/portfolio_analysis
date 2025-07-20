#!/usr/bin/env python3
"""Comprehensive portfolio analysis tool with PDF report generation.

This module provides advanced portfolio analytics including:
- Portfolio composition and weight analysis
- Time series analysis of holdings
- Risk and return metrics
- Performance attribution
- Portfolio optimization
- Custom tearsheet generation
"""

import glob
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PortfolioAnalyzer:
    """Comprehensive portfolio analysis with PDF report generation."""
    
    def __init__(self, years_back: int = 5):
        """Initialize the portfolio analyzer.
        
        Args:
            years_back: Number of years of historical data to analyze
        """
        self.years_back = years_back
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=years_back * 365)
        
        # Data containers
        self.portfolio_df: Optional[pd.DataFrame] = None
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.returns_data: pd.DataFrame = pd.DataFrame()
        self.portfolio_weights: Dict[str, float] = {}
        self.risk_free_rate: Optional[pd.Series] = None
        self.spy_data: Optional[pd.DataFrame] = None
        
        # Analysis results
        self.portfolio_metrics: Dict = {}
        self.ratios_df: pd.DataFrame = pd.DataFrame()
        self.alpha_beta_df: pd.DataFrame = pd.DataFrame()
        self.optimal_weights: Dict[str, float] = {}
        
        print(f"📊 Portfolio Analyzer initialized for {years_back} years")
        print(f"   Analysis period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def find_latest_csv(self) -> str:
        """Find the most recent portfolio holdings CSV file.
        
        Returns:
            Path to the most recent CSV file
            
        Raises:
            FileNotFoundError: If no portfolio CSV files are found
        """
        pattern = "portfolio_holdings_*.csv"
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No portfolio CSV files found matching pattern: {pattern}")
        
        # Sort by modification time and get the most recent
        latest_file = max(csv_files, key=os.path.getmtime)
        print(f"📁 Using portfolio file: {latest_file}")
        return latest_file
    
    def load_portfolio_data(self) -> pd.DataFrame:
        """Load and process portfolio data from the latest CSV file.
        
        Returns:
            Processed DataFrame with combined holdings
        """
        csv_file = self.find_latest_csv()
        df = pd.read_csv(csv_file)
        
        print(f"📈 Loaded {len(df)} total holdings")
        
        # Filter out holdings without tickers
        df_with_tickers = df[df['Ticker'].notna() & (df['Ticker'] != '')].copy()
        print(f"🎯 {len(df_with_tickers)} holdings have valid tickers")
        
        # Exclude Grand total row
        df_with_tickers = df_with_tickers[df_with_tickers['Holding'] != 'Grand total'].copy()
        
        # Convert numeric columns
        numeric_columns = ['Shares', 'Price', 'Change', 'Value']
        for col in numeric_columns:
            if col in df_with_tickers.columns:
                # Remove currency symbols and commas, convert to float
                df_with_tickers[col] = df_with_tickers[col].astype(str).str.replace('[$,+]', '', regex=True)
                df_with_tickers[col] = pd.to_numeric(df_with_tickers[col], errors='coerce')
        
        # Combine duplicate tickers
        df_combined = self.combine_duplicate_tickers(df_with_tickers)
        
        self.portfolio_df = df_combined
        print(f"✅ Final portfolio: {len(df_combined)} unique tickers")
        return df_combined
    
    def combine_duplicate_tickers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine holdings with the same ticker symbol.
        
        Args:
            df: DataFrame with individual holdings
            
        Returns:
            DataFrame with combined holdings by ticker
        """
        combined_data = []
        
        for ticker in df['Ticker'].unique():
            ticker_rows = df[df['Ticker'] == ticker]
            
            if len(ticker_rows) == 1:
                combined_data.append(ticker_rows.iloc[0])
            else:
                # Combine multiple holdings of the same ticker
                combined_row = ticker_rows.iloc[0].copy()
                combined_row['Shares'] = ticker_rows['Shares'].sum()
                combined_row['Value'] = ticker_rows['Value'].sum()
                
                # Weight-average price
                total_shares = ticker_rows['Shares'].sum()
                if total_shares > 0:
                    combined_row['Price'] = ticker_rows['Value'].sum() / total_shares
                
                # Update holding name to indicate combination
                holding_names = ticker_rows['Holding'].tolist()
                if len(holding_names) > 1:
                    combined_row['Holding'] = f"{ticker} (Combined: {len(holding_names)} positions)"
                
                combined_data.append(combined_row)
                print(f"🔄 Combined {len(ticker_rows)} positions for {ticker}")
        
        return pd.DataFrame(combined_data).reset_index(drop=True)
    
    def download_price_data(self) -> Dict[str, pd.DataFrame]:
        """Download historical price data for all tickers.
        
        Returns:
            Dictionary mapping tickers to their price DataFrames
        """
        if self.portfolio_df is None:
            raise ValueError("Portfolio data not loaded. Call load_portfolio_data() first.")
        
        tickers = self.portfolio_df['Ticker'].tolist()
        print(f"\n💾 Downloading price data for {len(tickers)} tickers...")
        
        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"   {i:2d}/{len(tickers)}: {ticker}", end=" ")
                
                # Download data with yfinance
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)
                
                if not hist.empty:
                    # Use adjusted close prices - check available columns
                    if 'Adj Close' in hist.columns:
                        price_col = 'Adj Close'
                    elif 'Close' in hist.columns:
                        price_col = 'Close'
                    else:
                        # Use the last column if neither is available
                        price_col = hist.columns[-1]
                    
                    price_df = hist[[price_col]].copy()
                    price_df.columns = [ticker]
                    self.price_data[ticker] = price_df
                    print(f"✅ ({len(hist)} days)")
                else:
                    print("❌ No data")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"📊 Successfully downloaded data for {len(self.price_data)} tickers")
        return self.price_data
    
    def get_risk_free_rate(self) -> pd.Series:
        """Download 10-year Treasury rate as risk-free rate.
        
        Returns:
            Series with risk-free rate data
        """
        try:
            print("📈 Downloading 10-year Treasury rate...")
            
            # Try to get Treasury data from FRED
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(start=self.start_date, end=self.end_date)
            
            if not hist.empty:
                # Check available columns and convert percentage to decimal
                if 'Adj Close' in hist.columns:
                    price_col = 'Adj Close'
                elif 'Close' in hist.columns:
                    price_col = 'Close'
                else:
                    price_col = hist.columns[-1]
                
                risk_free = hist[price_col] / 100
                self.risk_free_rate = risk_free
                print(f"✅ Downloaded Treasury data ({len(risk_free)} days)")
                return risk_free
            else:
                # Fallback to a constant rate
                print("⚠️  Using fallback risk-free rate of 4.5%")
                dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
                self.risk_free_rate = pd.Series(0.045, index=dates)
                return self.risk_free_rate
                
        except Exception as e:
            print(f"⚠️  Error downloading Treasury data: {e}")
            print("   Using fallback risk-free rate of 4.5%")
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            self.risk_free_rate = pd.Series(0.045, index=dates)
            return self.risk_free_rate
    
    def download_spy_data(self) -> pd.DataFrame:
        """Download SPY data for benchmark analysis.
        
        Returns:
            DataFrame with SPY price data
        """
        try:
            print("📈 Downloading SPY benchmark data...")
            spy = yf.Ticker("SPY")
            hist = spy.history(start=self.start_date, end=self.end_date)
            
            if not hist.empty:
                # Check available columns
                if 'Adj Close' in hist.columns:
                    price_col = 'Adj Close'
                elif 'Close' in hist.columns:
                    price_col = 'Close'
                else:
                    price_col = hist.columns[-1]
                
                self.spy_data = hist[[price_col]].copy()
                self.spy_data.columns = ['SPY']
                print(f"✅ Downloaded SPY data ({len(hist)} days)")
                return self.spy_data
            else:
                raise ValueError("No SPY data available")
                
        except Exception as e:
            print(f"❌ Error downloading SPY data: {e}")
            raise
    
    def calculate_portfolio_weights(self) -> Dict[str, float]:
        """Calculate portfolio weights based on current values.
        
        Returns:
            Dictionary mapping tickers to their portfolio weights
        """
        if self.portfolio_df is None:
            raise ValueError("Portfolio data not loaded")
        
        total_value = self.portfolio_df['Value'].sum()
        weights = {}
        
        for _, row in self.portfolio_df.iterrows():
            weight = row['Value'] / total_value
            weights[row['Ticker']] = weight
        
        self.portfolio_weights = weights
        print(f"💰 Portfolio total value: ${total_value:,.2f}")
        return weights
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns for all holdings and the portfolio.
        
        Returns:
            DataFrame with daily returns for each ticker and portfolio
        """
        if not self.price_data:
            raise ValueError("Price data not available. Call download_price_data() first.")
        
        print("📊 Calculating returns...")
        
        # Combine all price data into a single DataFrame
        price_df = pd.DataFrame()
        for ticker, data in self.price_data.items():
            if not data.empty:
                price_df[ticker] = data.iloc[:, 0]  # First column is the price
        
        # Calculate daily returns
        returns_df = price_df.pct_change().dropna()
        
        # Calculate portfolio returns using weights
        if self.portfolio_weights:
            portfolio_returns = pd.Series(0.0, index=returns_df.index)
            for ticker, weight in self.portfolio_weights.items():
                if ticker in returns_df.columns:
                    portfolio_returns += returns_df[ticker] * weight
            
            returns_df['Portfolio'] = portfolio_returns
        
        self.returns_data = returns_df
        print(f"✅ Calculated returns for {len(returns_df.columns)} assets")
        return returns_df
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate comprehensive portfolio metrics.
        
        Returns:
            Dictionary with portfolio risk and return metrics
        """
        if self.returns_data.empty:
            raise ValueError("Returns data not available")
        
        print("📊 Calculating portfolio metrics...")
        
        returns = self.returns_data
        portfolio_returns = returns['Portfolio'] if 'Portfolio' in returns.columns else None
        
        if portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated")
        
        # Basic return metrics
        daily_return = portfolio_returns.mean()
        annual_return = daily_return * 252  # Assuming 252 trading days
        cumulative_return = (1 + portfolio_returns).prod() - 1
        
        # Risk metrics
        daily_vol = portfolio_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Portfolio covariance matrix
        asset_returns = returns.drop(columns=['Portfolio'], errors='ignore')
        cov_matrix = asset_returns.cov() * 252  # Annualized
        
        # Portfolio variance using weights
        portfolio_var = 0
        for i, ticker_i in enumerate(asset_returns.columns):
            for j, ticker_j in enumerate(asset_returns.columns):
                weight_i = self.portfolio_weights.get(ticker_i, 0)
                weight_j = self.portfolio_weights.get(ticker_j, 0)
                portfolio_var += weight_i * weight_j * cov_matrix.iloc[i, j]
        
        # Sharpe ratio
        risk_free_mean = self.risk_free_rate.mean() if self.risk_free_rate is not None else 0.045
        sharpe_ratio = (annual_return - risk_free_mean) / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol
        sortino_ratio = (annual_return - risk_free_mean) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative_value = (1 + portfolio_returns).cumprod()
        running_max = cumulative_value.expanding().max()
        drawdown = (cumulative_value - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'cumulative_return': cumulative_return,
            'portfolio_variance': portfolio_var,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'covariance_matrix': cov_matrix,
            'total_return': cumulative_return
        }
        
        self.portfolio_metrics = metrics
        print("✅ Portfolio metrics calculated")
        return metrics
    
    def calculate_sharpe_sortino_ratios(self) -> pd.DataFrame:
        """Calculate Sharpe and Sortino ratios for each asset.
        
        Returns:
            DataFrame with ratios for each asset
        """
        if self.returns_data.empty:
            raise ValueError("Returns data not available")
        
        print("📊 Calculating Sharpe and Sortino ratios...")
        
        returns = self.returns_data.drop(columns=['Portfolio'], errors='ignore')
        risk_free_mean = self.risk_free_rate.mean() if self.risk_free_rate is not None else 0.045
        
        ratios_data = []
        
        for ticker in returns.columns:
            asset_returns = returns[ticker].dropna()
            
            if len(asset_returns) < 30:  # Minimum data requirement
                continue
            
            # Annualized metrics
            annual_return = asset_returns.mean() * 252
            annual_vol = asset_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe = (annual_return - risk_free_mean) / annual_vol if annual_vol > 0 else 0
            
            # Sortino ratio
            downside_returns = asset_returns[asset_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol
            sortino = (annual_return - risk_free_mean) / downside_vol if downside_vol > 0 else 0
            
            ratios_data.append({
                'Ticker': ticker,
                'Annual_Return': annual_return,
                'Annual_Volatility': annual_vol,
                'Sharpe_Ratio': sharpe,
                'Sortino_Ratio': sortino
            })
        
        ratios_df = pd.DataFrame(ratios_data)
        self.ratios_df = ratios_df
        print(f"✅ Calculated ratios for {len(ratios_df)} assets")
        return ratios_df
    
    def calculate_alpha_beta(self) -> pd.DataFrame:
        """Calculate alpha and beta for each asset relative to SPY.
        
        Returns:
            DataFrame with alpha and beta values
        """
        if self.returns_data.empty or self.spy_data is None:
            raise ValueError("Returns data or SPY data not available")
        
        print("📊 Calculating alpha and beta vs SPY...")
        
        # Calculate SPY returns
        spy_returns = self.spy_data.pct_change().dropna()
        spy_returns.columns = ['SPY_Returns']
        
        returns = self.returns_data.drop(columns=['Portfolio'], errors='ignore')
        alpha_beta_data = []
        
        for ticker in returns.columns:
            asset_returns = returns[ticker].dropna()
            
            # Align dates
            common_dates = asset_returns.index.intersection(spy_returns.index)
            if len(common_dates) < 30:
                continue
            
            asset_aligned = asset_returns.loc[common_dates]
            spy_aligned = spy_returns.loc[common_dates, 'SPY_Returns']
            
            # Calculate beta using linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(spy_aligned, asset_aligned)
            beta = slope
            
            # Calculate alpha (annualized)
            alpha = intercept * 252
            
            alpha_beta_data.append({
                'Ticker': ticker,
                'Alpha': alpha,
                'Beta': beta,
                'R_Squared': r_value ** 2,
                'P_Value': p_value
            })
        
        alpha_beta_df = pd.DataFrame(alpha_beta_data)
        
        # Calculate portfolio alpha and beta
        if 'Portfolio' in self.returns_data.columns:
            portfolio_returns = self.returns_data['Portfolio'].dropna()
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            
            if len(common_dates) >= 30:
                portfolio_aligned = portfolio_returns.loc[common_dates]
                spy_aligned = spy_returns.loc[common_dates, 'SPY_Returns']
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(spy_aligned, portfolio_aligned)
                
                portfolio_row = {
                    'Ticker': 'Portfolio',
                    'Alpha': intercept * 252,
                    'Beta': slope,
                    'R_Squared': r_value ** 2,
                    'P_Value': p_value
                }
                alpha_beta_df = pd.concat([alpha_beta_df, pd.DataFrame([portfolio_row])], ignore_index=True)
        
        self.alpha_beta_df = alpha_beta_df
        print(f"✅ Calculated alpha/beta for {len(alpha_beta_df)} assets")
        return alpha_beta_df
    
    def optimize_portfolio(self) -> Dict[str, float]:
        """Find optimal portfolio allocation using mean-variance optimization.
        
        Returns:
            Dictionary with optimal weights
        """
        if self.returns_data.empty:
            raise ValueError("Returns data not available")
        
        print("🎯 Optimizing portfolio allocation...")
        
        returns = self.returns_data.drop(columns=['Portfolio'], errors='ignore')
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: long positions only (0 <= weights <= 0.05) with 5% max position constraint
        bounds = tuple((0, 0.05) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = dict(zip(returns.columns, result.x))
            
            # Filter out very small weights
            optimal_weights = {k: v for k, v in optimal_weights.items() if v > 0.001}
            
            # Renormalize
            total_weight = sum(optimal_weights.values())
            optimal_weights = {k: v/total_weight for k, v in optimal_weights.items()}
            
            self.optimal_weights = optimal_weights
            print(f"✅ Optimization successful. {len(optimal_weights)} assets in optimal portfolio")
            return optimal_weights
        else:
            print("❌ Optimization failed")
            return {}
    
    # Visualization Methods
    def plot_portfolio_composition(self) -> plt.Figure:
        """Create portfolio composition visualization.
        
        Returns:
            Figure with portfolio weights table and pie chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Create weights DataFrame for table
        weights_data = []
        for ticker, weight in sorted(self.portfolio_weights.items(), key=lambda x: x[1], reverse=True):
            holding_info = self.portfolio_df[self.portfolio_df['Ticker'] == ticker].iloc[0]
            weights_data.append({
                'Ticker': ticker,
                'Weight': f"{weight:.2%}",
                'Value': f"${holding_info['Value']:,.0f}",
                'Shares': f"{holding_info['Shares']:,.1f}"
            })
        
        weights_df = pd.DataFrame(weights_data)
        
        # Table
        ax1.axis('tight')
        ax1.axis('off')
        table = ax1.table(cellText=weights_df.values, colLabels=weights_df.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax1.set_title('Portfolio Holdings', fontsize=14, fontweight='bold', pad=20)
        
        # Pie chart (top 10 holdings + others)
        sorted_weights = sorted(self.portfolio_weights.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_weights[:10]
        others_weight = sum(weight for _, weight in sorted_weights[10:])
        
        if others_weight > 0:
            pie_data = [weight for _, weight in top_10] + [others_weight]
            pie_labels = [ticker for ticker, _ in top_10] + ['Others']
        else:
            pie_data = [weight for _, weight in top_10]
            pie_labels = [ticker for ticker, _ in top_10]
        
        ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Portfolio Composition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_holding_values_timeseries(self) -> plt.Figure:
        """Create time series plot of holding values.
        
        Returns:
            Figure with time series of holding values
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Calculate historical values for each holding
        value_data = pd.DataFrame()
        
        for ticker, price_df in self.price_data.items():
            if ticker in self.portfolio_weights:
                holding_info = self.portfolio_df[self.portfolio_df['Ticker'] == ticker].iloc[0]
                shares = holding_info['Shares']
                
                # Calculate historical values
                historical_values = price_df.iloc[:, 0] * shares
                value_data[ticker] = historical_values
        
        # Plot individual holdings (top 10 by weight)
        top_tickers = sorted(self.portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for ticker, _ in top_tickers:
            if ticker in value_data.columns:
                ax1.plot(value_data.index, value_data[ticker], label=ticker, linewidth=2)
        
        ax1.set_title('Individual Holding Values Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot total portfolio value
        portfolio_value = value_data.sum(axis=1)
        ax2.plot(portfolio_value.index, portfolio_value, linewidth=3, color='navy', label='Total Portfolio')
        ax2.fill_between(portfolio_value.index, portfolio_value, alpha=0.3, color='lightblue')
        
        ax2.set_title('Total Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Total Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_returns_boxplot(self) -> plt.Figure:
        """Create box plot of returns ordered by standard deviation.
        
        Returns:
            Figure with returns box plot
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        returns = self.returns_data.drop(columns=['Portfolio'], errors='ignore')
        
        # Calculate standard deviations and sort
        std_devs = returns.std().sort_values(ascending=True)
        sorted_tickers = std_devs.index.tolist()
        
        # Prepare data for box plot
        box_data = [returns[ticker].dropna() * 100 for ticker in sorted_tickers]  # Convert to percentage
        
        bp = ax.boxplot(box_data, labels=sorted_tickers, patch_artist=True)
        
        # Color boxes by standard deviation
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Daily Returns Distribution (Ordered by Volatility)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets (Low to High Volatility)')
        ax.set_ylabel('Daily Returns (%)')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def plot_risk_adjusted_returns(self) -> plt.Figure:
        """Create box plot of risk-adjusted returns.
        
        Returns:
            Figure with risk-adjusted returns plot
        """
        if self.ratios_df.empty:
            self.calculate_sharpe_sortino_ratios()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Sort by Sharpe ratio
        sorted_df = self.ratios_df.sort_values('Sharpe_Ratio', ascending=True)
        
        x_pos = np.arange(len(sorted_df))
        
        # Create bar plot
        bars = ax.bar(x_pos, sorted_df['Sharpe_Ratio'], 
                     color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_df))))
        
        ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Assets (Low to High Sharpe Ratio)')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_df['Ticker'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def create_pie_charts(self) -> plt.Figure:
        """Create pie charts for weights, risk, and return contributions.
        
        Returns:
            Figure with three pie charts
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Portfolio weights pie chart
        sorted_weights = sorted(self.portfolio_weights.items(), key=lambda x: x[1], reverse=True)
        top_8 = sorted_weights[:8]
        others_weight = sum(weight for _, weight in sorted_weights[8:])
        
        if others_weight > 0:
            weights_data = [weight for _, weight in top_8] + [others_weight]
            weights_labels = [ticker for ticker, _ in top_8] + ['Others']
        else:
            weights_data = [weight for _, weight in top_8]
            weights_labels = [ticker for ticker, _ in top_8]
        
        axes[0].pie(weights_data, labels=weights_labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('Portfolio Weights', fontsize=12, fontweight='bold')
        
        # Risk contribution (simplified as weight * volatility)
        if not self.ratios_df.empty:
            risk_contrib = {}
            for _, row in self.ratios_df.iterrows():
                ticker = row['Ticker']
                if ticker in self.portfolio_weights:
                    weight = self.portfolio_weights[ticker]
                    volatility = row['Annual_Volatility']
                    risk_contrib[ticker] = weight * volatility
            
            if risk_contrib:
                total_risk = sum(risk_contrib.values())
                risk_contrib = {k: v/total_risk for k, v in risk_contrib.items()}
                
                sorted_risk = sorted(risk_contrib.items(), key=lambda x: x[1], reverse=True)[:8]
                risk_data = [contrib for _, contrib in sorted_risk]
                risk_labels = [ticker for ticker, _ in sorted_risk]
                
                axes[1].pie(risk_data, labels=risk_labels, autopct='%1.1f%%', startangle=90)
        
        axes[1].set_title('Risk Contribution', fontsize=12, fontweight='bold')
        
        # Return contribution (weight * return)
        if not self.ratios_df.empty:
            return_contrib = {}
            for _, row in self.ratios_df.iterrows():
                ticker = row['Ticker']
                if ticker in self.portfolio_weights:
                    weight = self.portfolio_weights[ticker]
                    annual_return = row['Annual_Return']
                    return_contrib[ticker] = weight * annual_return
            
            if return_contrib:
                # Handle negative returns
                positive_contrib = {k: max(0, v) for k, v in return_contrib.items()}
                total_positive = sum(positive_contrib.values())
                
                if total_positive > 0:
                    positive_contrib = {k: v/total_positive for k, v in positive_contrib.items()}
                    sorted_returns = sorted(positive_contrib.items(), key=lambda x: x[1], reverse=True)[:8]
                    return_data = [contrib for _, contrib in sorted_returns]
                    return_labels = [ticker for ticker, _ in sorted_returns]
                    
                    axes[2].pie(return_data, labels=return_labels, autopct='%1.1f%%', startangle=90)
        
        axes[2].set_title('Return Contribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_pairs_plot(self) -> plt.Figure:
        """Create correlation pairs plot of returns.
        
        Returns:
            Figure with correlation heatmap
        """
        returns = self.returns_data.drop(columns=['Portfolio'], errors='ignore')
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        ax.set_title('Asset Returns Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


class PDFReportGenerator:
    """Handles PDF report generation with matplotlib figures and tables."""
    
    def __init__(self, filename: str):
        """Initialize PDF generator.
        
        Args:
            filename: Output PDF filename
        """
        self.filename = filename
        self.pdf_pages = None
        
    def __enter__(self):
        """Context manager entry."""
        self.pdf_pages = PdfPages(self.filename)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.pdf_pages:
            self.pdf_pages.close()
    
    def add_title_page(self, title: str, subtitle: str = "", metadata: Dict = None):
        """Add a title page to the PDF.
        
        Args:
            title: Main title
            subtitle: Subtitle text
            metadata: Dictionary with additional information
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, title, fontsize=24, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes)
        
        # Subtitle
        if subtitle:
            ax.text(0.5, 0.7, subtitle, fontsize=16, 
                    ha='center', va='center', transform=ax.transAxes)
        
        # Metadata
        if metadata:
            y_pos = 0.5
            for key, value in metadata.items():
                ax.text(0.5, y_pos, f"{key}: {value}", fontsize=12,
                        ha='center', va='center', transform=ax.transAxes)
                y_pos -= 0.05
        
        # Date
        ax.text(0.5, 0.1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=10, ha='center', va='center', transform=ax.transAxes)
        
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def add_figure(self, fig, title: str = ""):
        """Add a matplotlib figure to the PDF.
        
        Args:
            fig: Matplotlib figure object
            title: Optional title for the page
        """
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        self.pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def get_user_input() -> int:
    """Get number of years for analysis from user.
    
    Returns:
        Number of years to analyze
    """
    while True:
        try:
            years = input("\n📅 Enter number of years to analyze (default 5): ").strip()
            if not years:
                return 5
            years_int = int(years)
            if years_int < 1 or years_int > 20:
                print("⚠️  Please enter a number between 1 and 20")
                continue
            return years_int
        except ValueError:
            print("⚠️  Please enter a valid number")


def main() -> None:
    """Main execution function."""
    print("📊 Portfolio Analysis Tool")
    print("=" * 50)
    
    try:
        # Get user input
        years_back = get_user_input()
        
        # Initialize analyzer
        analyzer = PortfolioAnalyzer(years_back=years_back)
        
        # Load and process data
        print("\n🔄 Loading portfolio data...")
        analyzer.load_portfolio_data()
        
        print("\n🔄 Downloading price data...")
        analyzer.download_price_data()
        analyzer.get_risk_free_rate()
        analyzer.download_spy_data()
        
        # Calculate all metrics
        print("\n🔄 Calculating portfolio metrics...")
        analyzer.calculate_portfolio_weights()
        analyzer.calculate_returns()
        analyzer.calculate_portfolio_metrics()
        analyzer.calculate_sharpe_sortino_ratios()
        analyzer.calculate_alpha_beta()
        analyzer.optimize_portfolio()
        
        # Generate PDF report
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf_filename = f"portfolio_analysis_{timestamp}.pdf"
        
        print(f"\n📄 Generating PDF report: {pdf_filename}")
        
        with PDFReportGenerator(pdf_filename) as pdf:
            # Title page
            metadata = {
                'Analysis Period': f"{analyzer.start_date.strftime('%Y-%m-%d')} to {analyzer.end_date.strftime('%Y-%m-%d')}",
                'Number of Holdings': len(analyzer.portfolio_df),
                'Total Portfolio Value': f"${sum(analyzer.portfolio_df['Value']):,.2f}",
                'Assets with Price Data': len(analyzer.price_data)
            }
            pdf.add_title_page("Portfolio Analysis Report", 
                             f"Comprehensive Analysis - {years_back} Years", metadata)
            
            # Executive Summary
            if analyzer.portfolio_metrics:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Portfolio metrics table
                metrics = analyzer.portfolio_metrics
                summary_data = [
                    ['Annual Return', f"{metrics['annual_return']:.2%}"],
                    ['Annual Volatility', f"{metrics['annual_volatility']:.2%}"],
                    ['Sharpe Ratio', f"{metrics['sharpe_ratio']:.3f}"],
                    ['Sortino Ratio', f"{metrics['sortino_ratio']:.3f}"],
                    ['Max Drawdown', f"{metrics['max_drawdown']:.2%}"],
                    ['Cumulative Return', f"{metrics['cumulative_return']:.2%}"]
                ]
                
                ax1.axis('tight')
                ax1.axis('off')
                table1 = ax1.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                                  cellLoc='center', loc='center')
                table1.auto_set_font_size(False)
                table1.set_fontsize(11)
                table1.scale(1.2, 2)
                ax1.set_title('Portfolio Metrics', fontsize=14, fontweight='bold')
                
                # Top holdings table
                top_holdings = analyzer.portfolio_df.nlargest(10, 'Value')[['Ticker', 'Value']]
                top_holdings['Weight'] = top_holdings['Value'] / top_holdings['Value'].sum()
                top_holdings_data = []
                for _, row in top_holdings.iterrows():
                    top_holdings_data.append([row['Ticker'], 
                                            f"${row['Value']:,.0f}", 
                                            f"{row['Weight']:.1%}"])
                
                ax2.axis('tight')
                ax2.axis('off')
                table2 = ax2.table(cellText=top_holdings_data, 
                                  colLabels=['Ticker', 'Value', 'Weight'],
                                  cellLoc='center', loc='center')
                table2.auto_set_font_size(False)
                table2.set_fontsize(10)
                table2.scale(1.2, 1.5)
                ax2.set_title('Top 10 Holdings', fontsize=14, fontweight='bold')
                
                # Sharpe ratios chart
                if not analyzer.ratios_df.empty:
                    top_sharpe = analyzer.ratios_df.nlargest(8, 'Sharpe_Ratio')
                    ax3.barh(range(len(top_sharpe)), top_sharpe['Sharpe_Ratio'])
                    ax3.set_yticks(range(len(top_sharpe)))
                    ax3.set_yticklabels(top_sharpe['Ticker'])
                    ax3.set_xlabel('Sharpe Ratio')
                    ax3.set_title('Top Sharpe Ratios', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                
                # Optimal vs Current weights
                if analyzer.optimal_weights:
                    current_weights = list(analyzer.portfolio_weights.values())[:5]
                    current_tickers = list(analyzer.portfolio_weights.keys())[:5]
                    optimal_values = [analyzer.optimal_weights.get(t, 0) for t in current_tickers]
                    
                    x = np.arange(len(current_tickers))
                    width = 0.35
                    
                    ax4.bar(x - width/2, current_weights, width, label='Current', alpha=0.8)
                    ax4.bar(x + width/2, optimal_values, width, label='Optimal', alpha=0.8)
                    ax4.set_ylabel('Weight')
                    ax4.set_title('Current vs Optimal Allocation (Top 5)', fontsize=14, fontweight='bold')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(current_tickers, rotation=45)
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.add_figure(fig, "Executive Summary")
            
            # Portfolio composition
            print("   📊 Creating portfolio composition charts...")
            fig = analyzer.plot_portfolio_composition()
            pdf.add_figure(fig, "Portfolio Composition")
            
            # Time series analysis
            print("   📈 Creating time series plots...")
            fig = analyzer.plot_holding_values_timeseries()
            pdf.add_figure(fig, "Holdings Value Over Time")
            
            # Returns analysis
            print("   📊 Creating returns analysis...")
            fig = analyzer.plot_returns_boxplot()
            pdf.add_figure(fig, "Returns Distribution")
            
            fig = analyzer.plot_risk_adjusted_returns()
            pdf.add_figure(fig, "Risk-Adjusted Returns")
            
            # Risk analysis
            print("   🎯 Creating risk analysis...")
            fig = analyzer.create_pie_charts()
            pdf.add_figure(fig, "Risk and Return Attribution")
            
            fig = analyzer.create_pairs_plot()
            pdf.add_figure(fig, "Asset Correlation Analysis")
            
            # Performance tables
            print("   📋 Adding performance tables...")
            if not analyzer.ratios_df.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.axis('tight')
                ax.axis('off')
                
                # Format ratios table for display
                display_df = analyzer.ratios_df.copy()
                for col in ['Annual_Return', 'Annual_Volatility']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                for col in ['Sharpe_Ratio', 'Sortino_Ratio']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
                
                table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                               cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax.set_title('Asset Performance Metrics', fontsize=16, fontweight='bold', pad=20)
                
                pdf.add_figure(fig, "Detailed Performance Metrics")
            
            # Alpha/Beta analysis
            if not analyzer.alpha_beta_df.empty:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.axis('tight')
                ax.axis('off')
                
                # Format alpha/beta table
                display_df = analyzer.alpha_beta_df.copy()
                for col in ['Alpha']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                for col in ['Beta', 'R_Squared']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
                if 'P_Value' in display_df.columns:
                    display_df['P_Value'] = display_df['P_Value'].apply(lambda x: f"{x:.4f}")
                
                table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                               cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                ax.set_title('Alpha and Beta Analysis vs SPY', fontsize=16, fontweight='bold', pad=20)
                
                pdf.add_figure(fig, "Market Performance Analysis")
            
            # Portfolio Optimization Results
            print("   🎯 Adding optimization results...")
            if analyzer.optimal_weights:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Optimal weights table
                optimal_data = []
                total_optimal = sum(analyzer.optimal_weights.values())
                for ticker, weight in sorted(analyzer.optimal_weights.items(), key=lambda x: x[1], reverse=True):
                    current_weight = analyzer.portfolio_weights.get(ticker, 0)
                    optimal_data.append([
                        ticker,
                        f"{current_weight:.2%}",
                        f"{weight:.2%}",
                        f"{weight - current_weight:+.2%}"
                    ])
                
                ax1.axis('tight')
                ax1.axis('off')
                table = ax1.table(cellText=optimal_data, 
                                colLabels=['Ticker', 'Current', 'Optimal', 'Change'],
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax1.set_title('Optimal vs Current Allocation', fontsize=14, fontweight='bold')
                
                # Current vs Optimal weights comparison (bar chart)
                all_tickers = list(set(list(analyzer.portfolio_weights.keys()) + list(analyzer.optimal_weights.keys())))
                current_weights = [analyzer.portfolio_weights.get(t, 0) for t in all_tickers]
                optimal_weights_vals = [analyzer.optimal_weights.get(t, 0) for t in all_tickers]
                
                # Show top 10 by either current or optimal weight
                combined_weights = [(t, max(c, o)) for t, c, o in zip(all_tickers, current_weights, optimal_weights_vals)]
                top_tickers = sorted(combined_weights, key=lambda x: x[1], reverse=True)[:10]
                top_tickers_list = [t[0] for t in top_tickers]
                
                current_top = [analyzer.portfolio_weights.get(t, 0) for t in top_tickers_list]
                optimal_top = [analyzer.optimal_weights.get(t, 0) for t in top_tickers_list]
                
                x = np.arange(len(top_tickers_list))
                width = 0.35
                
                ax2.bar(x - width/2, current_top, width, label='Current', alpha=0.8, color='skyblue')
                ax2.bar(x + width/2, optimal_top, width, label='Optimal', alpha=0.8, color='lightcoral')
                ax2.set_ylabel('Weight')
                ax2.set_title('Current vs Optimal Allocation (Top 10)', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(top_tickers_list, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Weight changes (delta)
                changes = [(t, analyzer.optimal_weights.get(t, 0) - analyzer.portfolio_weights.get(t, 0)) 
                          for t in all_tickers]
                # Show biggest increases and decreases
                sorted_changes = sorted(changes, key=lambda x: abs(x[1]), reverse=True)[:10]
                
                change_tickers = [t[0] for t in sorted_changes]
                change_values = [t[1] for t in sorted_changes]
                colors = ['green' if v > 0 else 'red' for v in change_values]
                
                ax3.barh(range(len(change_tickers)), change_values, color=colors, alpha=0.7)
                ax3.set_yticks(range(len(change_tickers)))
                ax3.set_yticklabels(change_tickers)
                ax3.set_xlabel('Weight Change (Optimal - Current)')
                ax3.set_title('Largest Weight Changes', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='x')
                ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                
                # Optimization metrics comparison
                if analyzer.portfolio_metrics:
                    # Calculate expected metrics for optimal portfolio
                    returns_data = analyzer.returns_data.drop(columns=['Portfolio'], errors='ignore')
                    if not returns_data.empty:
                        # Expected return for optimal portfolio
                        expected_returns = returns_data.mean() * 252
                        optimal_return = sum(analyzer.optimal_weights.get(t, 0) * expected_returns.get(t, 0) 
                                           for t in expected_returns.index)
                        
                        # Expected volatility for optimal portfolio
                        cov_matrix = returns_data.cov() * 252
                        optimal_variance = 0
                        for i, ticker_i in enumerate(cov_matrix.index):
                            for j, ticker_j in enumerate(cov_matrix.columns):
                                weight_i = analyzer.optimal_weights.get(ticker_i, 0)
                                weight_j = analyzer.optimal_weights.get(ticker_j, 0)
                                optimal_variance += weight_i * weight_j * cov_matrix.iloc[i, j]
                        optimal_vol = np.sqrt(optimal_variance)
                        
                        # Risk-free rate for Sharpe calculation
                        risk_free_rate = analyzer.risk_free_rate.mean() if analyzer.risk_free_rate is not None else 0.045
                        optimal_sharpe = (optimal_return - risk_free_rate) / optimal_vol if optimal_vol > 0 else 0
                        
                        # Current portfolio metrics
                        current_return = analyzer.portfolio_metrics['annual_return']
                        current_vol = analyzer.portfolio_metrics['annual_volatility']
                        current_sharpe = analyzer.portfolio_metrics['sharpe_ratio']
                        
                        # Comparison table
                        comparison_data = [
                            ['Annual Return', f"{current_return:.2%}", f"{optimal_return:.2%}", 
                             f"{optimal_return - current_return:+.2%}"],
                            ['Annual Volatility', f"{current_vol:.2%}", f"{optimal_vol:.2%}", 
                             f"{optimal_vol - current_vol:+.2%}"],
                            ['Sharpe Ratio', f"{current_sharpe:.3f}", f"{optimal_sharpe:.3f}", 
                             f"{optimal_sharpe - current_sharpe:+.3f}"]
                        ]
                        
                        ax4.axis('tight')
                        ax4.axis('off')
                        comp_table = ax4.table(cellText=comparison_data, 
                                             colLabels=['Metric', 'Current', 'Optimal', 'Improvement'],
                                             cellLoc='center', loc='center')
                        comp_table.auto_set_font_size(False)
                        comp_table.set_fontsize(11)
                        comp_table.scale(1.3, 2)
                        ax4.set_title('Expected Performance Improvement', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                pdf.add_figure(fig, "Portfolio Optimization Results")
            else:
                # If optimization failed, create a simple message
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, 'Portfolio optimization could not be completed.\nThis may be due to insufficient historical data\nor numerical optimization issues.', 
                        ha='center', va='center', fontsize=14, transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.axis('off')
                ax.set_title('Portfolio Optimization Status', fontsize=16, fontweight='bold')
                pdf.add_figure(fig, "Portfolio Optimization Results")
        
        print(f"\n🎉 Analysis complete!")
        print(f"   📄 PDF report saved: {pdf_filename}")
        print(f"   📊 {len(analyzer.portfolio_df)} holdings analyzed")
        print(f"   💰 Total portfolio value: ${sum(analyzer.portfolio_df['Value']):,.2f}")
        
        if analyzer.portfolio_metrics:
            metrics = analyzer.portfolio_metrics
            print(f"   📈 Annual return: {metrics['annual_return']:.2%}")
            print(f"   📉 Annual volatility: {metrics['annual_volatility']:.2%}")
            print(f"   ⚡ Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        print(f"\n📁 Open {pdf_filename} to view the complete analysis report.")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        raise


if __name__ == "__main__":
    main()