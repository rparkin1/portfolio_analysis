# Empower Portfolio Analysis Suite

A comprehensive Python toolkit for scraping portfolio holdings data from Empower (formerly Personal Capital) and performing advanced portfolio analysis with optimization and reporting capabilities.

## Features

### Data Collection (`get_data.py`)
- **🏦 Web Scraping**: Automated data extraction from Empower portfolio holdings page
- **🔍 Ticker Lookup**: Intelligent ticker symbol identification and verification
- **✅ Yahoo Finance Integration**: Real-time ticker verification against market data
- **📊 CSV Export**: Clean, structured output with ticker information
- **🛡️ Error Handling**: Robust handling of login failures, page changes, and network issues
- **🔄 Single Process**: Everything runs in one streamlined command

### Portfolio Analysis (`analysis.py`)
- **📈 Performance Analysis**: Calculate returns, volatility, Sharpe/Sortino ratios
- **🎯 Portfolio Optimization**: Mean-variance optimization with 5% position size constraints
- **📊 Risk Metrics**: Alpha, beta, maximum drawdown, and risk attribution
- **📉 Visualizations**: Time series plots, correlation heatmaps, composition charts
- **🎨 Professional Reports**: Comprehensive PDF reports with all analysis and charts
- **⚖️ Benchmarking**: Performance comparison against S&P 500 (SPY)
- **🔄 Flexible Timeframes**: User-configurable analysis periods

## Requirements

- Python 3.8+
- Empower (Personal Capital) account
- Internet connection for Yahoo Finance API
- Supported browsers: Chrome, Firefox, or Safari

## Installation

1. **Clone or download the project**:
   ```bash
   cd /path/to/your/project
   ```

2. **Install dependencies using uv**:
   ```bash
   uv add selenium pandas webdriver-manager yfinance requests scipy seaborn plotly reportlab
   ```

   Or if you prefer pip:
   ```bash
   pip install selenium pandas webdriver-manager yfinance requests scipy seaborn plotly reportlab
   ```

3. **Ensure browser drivers are available** (webdriver-manager handles this automatically)

## Usage

### Data Collection

Run the scraper to collect portfolio holdings data:

```bash
uv run python get_data.py
```

### Portfolio Analysis

After collecting data, run comprehensive portfolio analysis:

```bash
uv run python analysis.py
```

### Complete Workflow

#### Step 1: Data Collection (`get_data.py`)
1. **Browser Selection**: Choose your preferred browser (Chrome, Firefox, Safari)
2. **Manual Login**: Complete login process in the opened browser window
3. **Data Scraping**: Automatic navigation and data extraction
4. **Ticker Lookup**: Intelligent ticker identification and verification
5. **CSV Export**: Final output with all data and tickers

#### Step 2: Portfolio Analysis (`analysis.py`)
1. **Data Loading**: Automatically finds most recent CSV file
2. **Timeframe Selection**: Choose analysis period (6 months, 1 year, 2 years, etc.)
3. **Price Data Download**: Fetch historical prices from Yahoo Finance
4. **Analysis Execution**: Calculate all metrics, ratios, and optimization
5. **PDF Report Generation**: Create comprehensive analysis report

### Example Output

#### Data Collection Output
```
🏦 Empower Portfolio Holdings Scraper with Ticker Lookup
============================================================
✅ Login successful!
✅ Holdings page loaded successfully!
✅ Scraped 40 holdings records

🔍 Looking up tickers for 40 holdings...
 1. IBM - International Business Machines Corp → ✅ IBM
 2. VGSH - Vanguard Short-Term Treasury → ✅ VGSH
 3. Cash → ✅ BIL (mapped)

✅ EXPORT COMPLETE
   File: portfolio_holdings_2025-07-20_14-30-15.csv
   Holdings with tickers: 40/40
```

#### Portfolio Analysis Output
```
📊 Portfolio Analysis Suite
============================================================

📂 Loading portfolio data from: portfolio_holdings_2025-07-20_14-30-15.csv
📈 Downloading price data for 35 assets over 1 year period...
🎯 Optimizing portfolio allocation with 5% position constraint...

📊 Analysis Results:
   Portfolio Return: 12.5% (annual)
   Portfolio Volatility: 8.2% (annual)
   Sharpe Ratio: 1.34
   Maximum Drawdown: -4.7%
   Alpha vs SPY: 2.1%
   Beta vs SPY: 0.87

✅ PDF Report Generated: portfolio_analysis_2025-07-20_15-45-30.pdf
```

## Output Format

### CSV Data File (`get_data.py` output)

The generated CSV file includes the following columns:

### Original Data Columns
- **Holding**: Security name/description
- **Shares**: Number of shares held
- **Price**: Current price per share
- **Change**: Price change
- **1 Day %**: One-day percentage change
- **1 day $**: One-day dollar change  
- **Value**: Total position value

### Added Ticker Columns
- **Ticker**: Verified ticker symbol
- **Ticker_Status**: How the ticker was determined

### Ticker Status Types
- `Verified existing ticker`: Ticker extracted from holding name and verified
- `User-provided mapping (verified)`: Custom mapping verified on Yahoo Finance
- `User mapping partial match`: Partial text match from custom mappings
- `Cash position`: Cash holdings (mapped to BIL)
- `401(k) plan-specific fund`: Internal fund codes without public tickers
- `Non-tradeable asset`: Employee stock plans, restricted shares, etc.

### PDF Analysis Report (`analysis.py` output)

The comprehensive PDF report includes:

#### 1. Portfolio Composition Analysis
- **Weights Table**: Current allocation percentages by ticker
- **Pie Chart**: Visual portfolio composition breakdown
- **Holdings Summary**: Value and share counts

#### 2. Performance Analysis
- **Time Series Charts**: Portfolio value over time vs benchmark
- **Returns Analysis**: Box plots of returns by volatility
- **Risk-Adjusted Returns**: Performance per unit of risk

#### 3. Risk Metrics
- **Correlation Heatmap**: Asset correlation matrix
- **Risk Contribution**: How each asset contributes to portfolio risk
- **Alpha/Beta Analysis**: Performance vs S&P 500 benchmark

#### 4. Portfolio Optimization Results
- **Optimal vs Current**: Side-by-side allocation comparison
- **Weight Changes**: Required rebalancing visualization
- **Expected Improvement**: Projected performance enhancement
- **5% Position Constraint**: No single asset exceeds 5% in optimal portfolio

#### 5. Key Metrics Summary
- Annual returns, volatility, Sharpe/Sortino ratios
- Maximum drawdown and recovery periods
- Portfolio alpha and beta coefficients

## Ticker Mapping Logic

The tool uses a multi-layered approach for ticker identification:

### 1. User-Provided Mappings (Highest Priority)
Pre-configured mappings for common holdings:
```python
'AMU03 - U.S. EQUITY INDEX': 'VTI'
'REIT IDX': 'VGSLX' 
'Cash': 'BIL'
'W W Grainger Inc': 'IBM'
# ... and many more
```

### 2. Pattern Extraction
Extracts tickers from formats like:
- `IBM - International Business Machines Corp` → `IBM`
- `VGSH - Vanguard Short-Term Treasury...` → `VGSH`

### 3. Yahoo Finance Verification
All tickers are verified against Yahoo Finance to ensure validity.

### 4. Special Cases
- 401(k) fund codes (AMU03, AMX01, etc.) → Mapped to equivalent ETFs
- Cash positions → Mapped to BIL (short-term treasury ETF)
- Employee stock plans → Mapped to underlying stock or marked as non-tradeable

## Troubleshooting

### Login Issues
- Ensure you complete the full login process including 2FA
- Wait for the page to fully load before pressing Enter
- Try refreshing the page if login detection fails

### Data Scraping Issues
- Check that you're on the correct holdings page
- Ensure the page has fully loaded before the script proceeds
- Review `debug_page_source.html` if generated for page structure analysis

### Ticker Lookup Issues
- Network connectivity required for Yahoo Finance API
- Some 401(k) funds may not have public ticker equivalents
- Custom mappings can be added to the `ticker_mappings` dictionary

### Browser Issues
- Update your browser to the latest version
- Clear browser cache and cookies
- Try a different browser if one fails

## Configuration

### Adding Custom Ticker Mappings

Edit the `ticker_mappings` dictionary in the `TickerLookup` class:

```python
self.ticker_mappings = {
    'Your Fund Name': 'TICKER',
    'Another Fund': 'SYMBOL',
    # ... existing mappings
}
```

### Modifying Wait Times

Adjust timing in the code if needed:
- Login wait time: Modify `time.sleep()` values in `login()` method
- Page load delays: Adjust waits in `scrape_holdings_data()` method
- Rate limiting: Change `time.sleep(0.3)` in ticker lookup loop

## File Structure

```
getdata/
├── get_data.py              # Data scraper with integrated ticker lookup
├── analysis.py              # Portfolio analysis and optimization tool
├── pyproject.toml           # Project dependencies and configuration
├── uv.lock                  # Dependency lock file
├── README.md                # This file
├── CLAUDE.md                # Development guidelines
├── portfolio_holdings_*.csv # Scraped data files (generated)
└── portfolio_analysis_*.pdf # Analysis reports (generated)
```

## Dependencies

### Data Collection Dependencies
- **selenium**: Web browser automation for scraping
- **pandas**: Data manipulation and CSV handling
- **webdriver-manager**: Automatic browser driver management
- **yfinance**: Yahoo Finance API for ticker verification and price data
- **requests**: HTTP requests for API calls

### Analysis Dependencies  
- **scipy**: Scientific computing and optimization algorithms
- **seaborn**: Statistical data visualization
- **plotly**: Interactive plotting capabilities
- **reportlab**: PDF report generation
- **matplotlib**: Core plotting functionality

## Security Notes

- The tool requires manual login to maintain security
- No credentials are stored or transmitted by the script
- All authentication is handled through the browser interface
- Rate limiting is applied to API calls to respect service limits

## Contributing

This project follows the guidelines in `CLAUDE.md`. Key points:

- Use `uv` for package management (never pip directly)
- Add type hints to all functions
- Follow PEP 8 naming conventions
- Test thoroughly with realistic data
- Keep functions focused and small

## License

This tool is for personal use only. Ensure compliance with Empower's terms of service when using this scraper.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated `debug_page_source.html` if available
3. Ensure all dependencies are correctly installed
4. Verify your Empower account access through the web interface

---

**Note**: This tool is designed for personal portfolio tracking and analysis. Always verify the accuracy of scraped data against your official account statements.