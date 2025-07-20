#!/usr/bin/env python3
"""Portfolio holdings data scraper for Empower with integrated ticker lookup.

This script scrapes portfolio holdings data from Empower (formerly Personal Capital)
and automatically looks up and verifies ticker symbols for each holding using:
1. User-provided ticker mappings for accurate fund/stock identification
2. Yahoo Finance API for ticker verification
3. Intelligent pattern matching for common holdings

The output CSV includes all scraped data plus Ticker and Ticker_Status columns.
"""

import csv
import getpass
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from requests.exceptions import RequestException


class TickerLookup:
    """Integrated ticker lookup with user-provided mappings."""
    
    def __init__(self) -> None:
        """Initialize the ticker lookup service."""
        self.ticker_cache: Dict[str, str] = {}
        self.invalid_tickers: set = set()
        
        # User-provided ticker replacement mapping
        self.ticker_mappings = {
            'NON40OJFF': 'FSPSX', 
            'NON40OJFE': 'FSKAX',
            'AMX04': 'JKD',
            'AMX02': 'FCBFX',
            '95RA8': 'BIL',
            '94SX0': 'BIL',
            'Cash': 'BIL',
            'EMPLOYER CONTRIBUTIONS ACCOUNT': 'BIL',
            'NON40OJFD': 'VNQ',
            'NON40OJFB': 'TIP',
            'LONG-TERM CORP BOND': 'IGLB',
            'INTERNATIONAL VALUE': 'FIVLX',
            'REIT Index': 'VNQ',
            'Emerging Markets': 'EEM',
            'INTERNATIONAL GROWTH': 'EFA',
            'SMALL MID CAP INDEX': 'VO',
            'BLACKROCK FEDFUND INSTL CL': 'TIP',
            'APLVX': 'VTV',
            'APMGX': 'VO', 
            'AB4908CIT': 'VTIVX',
            'VANGUARD INDEX FDS ETF': 'VOO',
            'INVESCO QQQ TR ETF': 'QQQ', 
            'AMU01': 'IGLB', 
            'AMU03': 'VOO', 
            'AMX01 - CAPITAL PRESERVATION': 'BIL', 
            'AMU05': 'FIVLX', 
            'AMU04': 'VOO', 
            'AIIIX': 'VTMGX',
            'NON40OJFA': 'PIMIX',
            'QWPAQ': 'BIL',
            'BRK.B': 'BRK-B',
            'Large-Cap Growth Index Fund': 'FSPGX',
            'Large Company Index Fund': 'VLCAX',
            'Total International Stock Market Index Fund': 'VTIAX',
            'Real Estate Investment Trust Index Fund': 'VGSLX',
            'Total Stock Market Index Fund': 'VTSAX',
            'European Stock Index Fund': 'VEUSX',
            'INFLATION PROTECTED BOND FUND': 'FIPDX',
            'Long-Term Corporate Bond Fund': 'IGLB',
            'TIPS': 'FIPDX',
            'TIP': 'FIPDX',
            'IBM Options': 'IBM',
            '2045 Retirement Portfolio': 'VTIVX',
            'EUROPEAN STOCK IDX': 'VEUSX',
            'Total Bond Market': 'FTBFX',
            'REIT IDX': 'VGSLX',
            'LARGE-CAP VALUE IDX': 'VVIAX',
            'LARGE COMPANY IDX': 'VLCAX',
            'INTL STOCK MKT IDX': 'VTIAX',
            'TOTAL STOCK MKT IDX': 'VTSAX',
            'LARGE-CAP GROWTH IDX': 'VIGAX',
            'INFL PROTECTED BOND': 'TIP',
            'W W Grainger Inc': 'IBM',
            'SMALL/MID-CAP IDX': 'BSMAX',
            'Other Stock Plan Shares - from Morgan Stanley Accounts': 'IBM',
            'AMU03 - U.S. EQUITY INDEX': 'VTI',
            'AMU05 - INTERNATIONAL EQUITY ACTIVE': 'VWICX',
            'AMU01 - FIXED INCOME INDEX': 'BIL',
            '94SX0 - ML DIRECT DEPOSIT PROGRM': 'BIL',
            'AMX02 - FIXED INCOME': 'BND',
            'AMU04 - U.S. EQUITY ACTIVE': 'VDEQX',
            'IBM@ - IBM Restricted Stock Units Vesting':'IBM'
        }
        
    def extract_ticker_from_holding(self, holding: str) -> Optional[str]:
        """Extract ticker symbol from holding name if present."""
        if not holding or holding in ["Cash", "Grand total"]:
            return None
            
        # Check user mappings first
        if holding in self.ticker_mappings:
            return self.ticker_mappings[holding]
            
        # Pattern 1: "TICKER - Description" format
        match = re.match(r'^([A-Z]{1,5}(?:\.[A-Z])?)\s*-\s*(.+)', holding)
        if match:
            ticker = match.group(1)
            # Exclude obvious non-tickers
            if ticker not in ["FIXED", "REIT", "LONG", "TOTAL", "LARGE", "SMALL", "INTL", "INFL"]:
                return ticker
            
        return None
    
    def verify_ticker_yahoo(self, ticker: str) -> bool:
        """Verify if ticker exists on Yahoo Finance."""
        if not ticker or ticker in self.invalid_tickers:
            return False
            
        if ticker in self.ticker_cache:
            return True
            
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if (info and 
                len(info) > 5 and
                ('symbol' in info or 'shortName' in info) and
                ('regularMarketPrice' in info or 'previousClose' in info)):
                
                self.ticker_cache[ticker] = ticker
                return True
            else:
                self.invalid_tickers.add(ticker)
                return False
                
        except Exception:
            self.invalid_tickers.add(ticker)
            return False
    
    def lookup_ticker(self, holding: str) -> Tuple[Optional[str], str]:
        """Look up ticker for a holding with comprehensive logic."""
        if not holding or holding.strip() in ["", "Grand total"]:
            return None, "Total row"
            
        # Check user mappings first (highest priority)
        if holding in self.ticker_mappings:
            mapped_ticker = self.ticker_mappings[holding]
            if self.verify_ticker_yahoo(mapped_ticker):
                return mapped_ticker, "User-provided mapping (verified)"
            else:
                return mapped_ticker, "User-provided mapping (unverified)"
        
        # Check for partial matches in user mappings
        for key, value in self.ticker_mappings.items():
            if key in holding or holding in key:
                if self.verify_ticker_yahoo(value):
                    return value, f"User mapping partial match: {key}"
                
        # Try to extract ticker from holding name
        extracted_ticker = self.extract_ticker_from_holding(holding)
        if extracted_ticker:
            if self.verify_ticker_yahoo(extracted_ticker):
                return extracted_ticker, "Verified existing ticker"
        
        # Handle special cases for cash and non-tradeable assets
        if holding == "Cash":
            return "BIL", "Cash mapped to BIL"
            
        # Special handling for fund codes
        if re.match(r'^[A-Z]{3}\d{2}', holding):
            return None, "401(k) plan-specific fund"
            
        return None, "No ticker found"


class EmpowerScraper:
    """Web scraper for Empower portfolio holdings data."""

    LOGIN_URL = (
        "https://home.personalcapital.com/page/login/goHome"
        "#_gl=1*gr0l9c*_gcl_au*MTE2ODk1ODM2LjE3NTE1NzY3OTI."
    )
    HOLDINGS_URL = "https://home.personalcapital.com/page/login/app#/portfolio/holdings"

    def __init__(self, browser: str = "chrome") -> None:
        """Initialize the scraper with browser choice.
        
        Args:
            browser: Browser to use ('chrome', 'firefox', or 'safari')
        """
        self.browser = browser.lower()
        self.driver: Optional[webdriver.Remote] = None
        self.wait: Optional[WebDriverWait] = None

    def setup_driver(self) -> webdriver.Remote:
        """Set up and return the WebDriver instance."""
        if self.browser == "chrome":
            options = webdriver.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)
            service = ChromeService(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        elif self.browser == "firefox":
            options = webdriver.FirefoxOptions()
            options.set_preference("dom.webdriver.enabled", False)
            options.set_preference("useAutomationExtension", False)
            service = FirefoxService(GeckoDriverManager().install())
            driver = webdriver.Firefox(service=service, options=options)
        elif self.browser == "safari":
            driver = webdriver.Safari()
        else:
            raise ValueError(f"Unsupported browser: {self.browser}")

        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        self.driver = driver
        self.wait = WebDriverWait(driver, 30)
        return driver

    def login(self) -> bool:
        """Handle user login to Empower.
        
        Returns:
            True if login appears successful, False otherwise
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized. Call setup_driver() first.")

        print("Opening Empower login page...")
        self.driver.get(self.LOGIN_URL)
        time.sleep(3)

        print("\nPlease complete the login process in the browser window.")
        print("Enter your email and password, complete any 2FA if required.")
        print("Once you're logged in and see the dashboard, press Enter here...")
        
        input("Press Enter after you've successfully logged in: ")

        current_url = self.driver.current_url
        print(f"Current URL: {current_url}")
        
        # More flexible login detection
        success_indicators = [
            "app" in current_url,
            "dashboard" in current_url,
            "home" in current_url and "login" not in current_url,
            "portfolio" in current_url
        ]
        
        if any(success_indicators):
            print("✅ Login successful!")
            return True
        
        # Still on login page - give user option to continue anyway
        if "login" in current_url.lower():
            print("⚠️  Still on login page.")
            user_confirm = input("Are you successfully logged in anyway? (y/n): ").lower().strip()
            if user_confirm in ['y', 'yes']:
                print("✅ Proceeding based on user confirmation.")
                return True
            else:
                print("Please complete login and run the script again.")
                return False
        
        # Unknown URL - ask user
        user_confirm = input("Login status unclear. Are you logged in? (y/n): ").lower().strip()
        if user_confirm in ['y', 'yes']:
            print("✅ Proceeding based on user confirmation.")
            return True
        else:
            return False

    def navigate_to_holdings(self) -> bool:
        """Navigate to the portfolio holdings page.
        
        Returns:
            True if navigation successful, False otherwise
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized.")

        print("Navigating to portfolio holdings page...")
        self.driver.get(self.HOLDINGS_URL)
        time.sleep(5)

        try:
            self.wait.until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            print("✅ Holdings page loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load holdings page: {e}")
            return False

    def scrape_holdings_data(self) -> List[Dict[str, str]]:
        """Scrape portfolio holdings data from the page.
        
        Returns:
            List of dictionaries containing holdings data
        """
        if not self.driver:
            raise RuntimeError("Driver not initialized.")

        print("Scraping holdings data...")
        time.sleep(5)  # Give more time for data to load

        # First, let's wait for data to actually load and debug what we see
        print("Waiting for page to fully load...")
        time.sleep(3)
        
        # Debug: Print current page info
        print(f"Current URL: {self.driver.current_url}")
        print(f"Page title: {self.driver.title}")
        
        try:
            # Target the specific Empower holdings table
            table = None
            
            # Look for the specific Empower holdings grid
            empower_selectors = [
                ".pc-holdings-grid.qa-datagrid-rows",
                "[role='grid'].pc-holdings-grid",
                ".table.pc-holdings-grid.qa-datagrid-rows",
                ".holdings-grid-table-client-container [role='grid']"
            ]
            
            for selector in empower_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        table = elements[0]
                        print(f"Found Empower table using selector: {selector}")
                        break
                except:
                    continue
            
            if not table:
                print("Empower table not found. Trying fallback selectors...")
                # Fallback to any role=grid
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, "[role='grid']")
                    if elements:
                        table = elements[0]
                        print("Found table using fallback [role='grid'] selector")
                except:
                    pass
            
            if not table:
                print("No table found. Trying alternative scraping...")
                return self._try_alternative_scraping()

            # Extract headers specifically for Empower structure
            headers = self._extract_empower_headers(table)
            print(f"Found headers: {headers}")
            
            if not headers or all(not h.strip() for h in headers):
                print("No valid headers found. Trying alternative methods...")
                return self._try_alternative_scraping()

            # Get table data with Empower-specific strategy
            holdings_data = self._extract_empower_data(table, headers)
            
            print(f"✅ Scraped {len(holdings_data)} holdings records")
            return holdings_data

        except Exception as e:
            print(f"❌ Error scraping data: {e}")
            return self._try_alternative_scraping()
    
    def _extract_headers(self, table) -> List[str]:
        """Extract headers from table using multiple strategies."""
        headers = []
        
        # Strategy 1: thead > tr > th
        try:
            thead = table.find_element(By.TAG_NAME, "thead")
            header_row = thead.find_element(By.TAG_NAME, "tr")
            for th in header_row.find_elements(By.TAG_NAME, "th"):
                headers.append(th.text.strip())
            if headers and any(h.strip() for h in headers):
                return headers
        except:
            pass
        
        # Strategy 2: first row as headers
        try:
            first_row = table.find_element(By.TAG_NAME, "tr")
            for cell in first_row.find_elements(By.CSS_SELECTOR, "th, td"):
                headers.append(cell.text.strip())
            if headers and any(h.strip() for h in headers):
                return headers
        except:
            pass
            
        # Strategy 3: look for header-like elements
        try:
            header_elements = table.find_elements(By.CSS_SELECTOR, 
                "[role='columnheader'], .header, .table-header, .column-header")
            for elem in header_elements:
                headers.append(elem.text.strip())
            if headers and any(h.strip() for h in headers):
                return headers
        except:
            pass
            
        return headers
    
    def _extract_empower_headers(self, table) -> List[str]:
        """Extract headers specifically from Empower table structure."""
        headers = []
        
        try:
            # Look for the header row with role="row" and class containing "header"
            header_row = table.find_element(By.CSS_SELECTOR, "[role='row'].table__row--header")
            
            # Get column headers with role="columnheader"
            column_headers = header_row.find_elements(By.CSS_SELECTOR, "[role='columnheader']")
            
            for header in column_headers:
                # Extract text from button or direct text
                try:
                    button = header.find_element(By.TAG_NAME, "button")
                    header_text = button.text.strip()
                except:
                    header_text = header.text.strip()
                
                headers.append(header_text)
            
            return headers
            
        except Exception as e:
            print(f"Error extracting Empower headers: {e}")
            return []
    
    def _extract_empower_data(self, table, headers: List[str]) -> List[Dict[str, str]]:
        """Extract data specifically from Empower table structure."""
        holdings_data = []
        
        try:
            # Find the table body with class containing "table__body"
            tbody = table.find_element(By.CSS_SELECTOR, "[role='rowgroup'].table__body")
            
            # Get all data rows (exclude header rows)
            data_rows = tbody.find_elements(By.CSS_SELECTOR, "[role='row']:not(.table__row--header)")
            
            for row in data_rows:
                # Get all grid cells
                cells = row.find_elements(By.CSS_SELECTOR, "[role='gridcell']")
                
                if len(cells) >= len(headers):
                    row_data = {}
                    for i, cell in enumerate(cells[:len(headers)]):
                        header = headers[i] if i < len(headers) else f"Column_{i}"
                        
                        # For Empower, extract text from various nested elements
                        cell_text = self._extract_cell_text(cell)
                        row_data[header] = cell_text
                    
                    # Only add rows with meaningful data
                    if any(row_data.values()) and row_data.get(headers[0], "").strip():
                        holdings_data.append(row_data)
            
            return holdings_data
            
        except Exception as e:
            print(f"Error extracting Empower data: {e}")
            return []
    
    def _extract_cell_text(self, cell) -> str:
        """Extract text from Empower table cell, handling nested structures."""
        # Check if this is a holding name cell
        if "holding-name" in cell.get_attribute("class"):
            try:
                # Get ticker symbol
                ticker_elem = cell.find_element(By.CSS_SELECTOR, ".qa-ticker")
                ticker = ticker_elem.text.strip()
                
                # Get description
                desc_elem = cell.find_element(By.CSS_SELECTOR, ".pc-holdings-grid-cell--holding-description")
                description = desc_elem.text.strip()
                
                # Combine ticker and description
                if ticker and description:
                    return f"{ticker} - {description}"
                elif ticker:
                    return ticker
                elif description:
                    return description
            except:
                pass
        
        # For numeric cells, look for span elements first
        try:
            span_elem = cell.find_element(By.TAG_NAME, "span")
            span_text = span_elem.text.strip()
            if span_text:
                return span_text
        except:
            pass
        
        # Fallback to direct cell text
        cell_text = cell.text.strip()
        
        # Clean up common formatting issues
        if cell_text:
            # Remove extra whitespace and newlines
            cell_text = ' '.join(cell_text.split())
            return cell_text
        
        return ""
    
    def _extract_table_data(self, table, headers: List[str]) -> List[Dict[str, str]]:
        """Extract data from table rows."""
        holdings_data = []
        
        # Try different row selectors
        row_selectors = ["tbody tr", "tr", "[role='row']"]
        
        for selector in row_selectors:
            try:
                rows = table.find_elements(By.CSS_SELECTOR, selector)
                if not rows:
                    continue
                    
                for row in rows:
                    cells = row.find_elements(By.CSS_SELECTOR, "td, th, [role='cell']")
                    if len(cells) >= len(headers) and len(cells) > 0:
                        row_data = {}
                        for i, cell in enumerate(cells[:len(headers)]):
                            header = headers[i] if i < len(headers) else f"Column_{i}"
                            cell_text = cell.text.strip()
                            row_data[header] = cell_text
                        
                        # Only add rows with meaningful data (not header rows)
                        if any(row_data.values()) and not all(v in headers for v in row_data.values()):
                            holdings_data.append(row_data)
                
                if holdings_data:
                    return holdings_data
                    
            except Exception as e:
                print(f"Error with selector {selector}: {e}")
                continue
                
        return holdings_data

    def _try_alternative_scraping(self) -> List[Dict[str, str]]:
        """Try alternative methods to scrape data if main method fails."""
        print("Trying alternative scraping methods...")
        
        # Save page source for debugging
        try:
            with open("debug_page_source.html", "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print("Page source saved to debug_page_source.html")
        except Exception as e:
            print(f"Could not save page source: {e}")
        
        # Strategy 1: Look for any text that might be holdings data
        holdings_data = []
        try:
            # Look for common patterns in financial data
            text_elements = self.driver.find_elements(By.CSS_SELECTOR, "*")
            
            # Try to find elements containing financial symbols or data patterns
            potential_data = []
            for elem in text_elements:
                text = elem.text.strip()
                if text and (
                    # Look for stock symbols (3-5 capital letters)
                    any(len(word) >= 3 and word.isupper() and word.isalpha() for word in text.split()) or
                    # Look for currency amounts
                    '$' in text or
                    # Look for percentages
                    '%' in text
                ):
                    potential_data.append(text)
            
            print(f"Found {len(potential_data)} potential data elements")
            for i, data in enumerate(potential_data[:10]):  # Show first 10
                print(f"  {i+1}: {data[:100]}...")
                
        except Exception as e:
            print(f"Text-based scraping failed: {e}")
        
        # Strategy 2: Try to wait longer and reload
        try:
            print("Trying to reload page and wait longer...")
            self.driver.refresh()
            time.sleep(10)
            
            # Look for any table-like structure again
            all_tables = self.driver.find_elements(By.CSS_SELECTOR, 
                "table, [role='table'], [role='grid'], .table, [class*='table'], [class*='grid']")
            
            print(f"Found {len(all_tables)} table-like elements after reload")
            
            for i, table in enumerate(all_tables):
                try:
                    print(f"Table {i+1} content preview: {table.text[:200]}...")
                except:
                    print(f"Table {i+1}: Could not read content")
            
        except Exception as e:
            print(f"Reload strategy failed: {e}")
        
        # Strategy 3: Manual user intervention
        print("\n" + "="*50)
        print("MANUAL INTERVENTION NEEDED")
        print("="*50)
        print("The automated scraping couldn't find the data.")
        print("Please help identify the data on the page:")
        print("1. Look at the browser window")
        print("2. The debug_page_source.html file has been saved")
        print("3. You can manually copy-paste data if needed")
        
        user_input = input("\nDo you see portfolio holdings data on the page? (y/n): ").lower().strip()
        if user_input in ['y', 'yes']:
            print("Please manually copy the data or help identify the correct selectors.")
            print("Check the debug_page_source.html file to find the right CSS selectors.")
        
        return holdings_data

    def export_to_csv(self, data: List[Dict[str, str]], filename: Optional[str] = None) -> str:
        """Export holdings data to CSV file with ticker lookup.
        
        Args:
            data: List of holdings dictionaries
            filename: Optional custom filename
            
        Returns:
            Path to the created CSV file
        """
        if not data:
            raise ValueError("No data to export")

        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"portfolio_holdings_{timestamp}.csv"

        # Create DataFrame from scraped data
        df = pd.DataFrame(data)
        
        # Add ticker lookup functionality
        print(f"\n🔍 Looking up tickers for {len(data)} holdings...")
        print("-" * 60)
        
        ticker_lookup = TickerLookup()
        tickers = []
        statuses = []
        
        for idx, row in df.iterrows():
            holding = str(row.get('Holding', ''))
            
            if not holding or holding == "Grand total":
                tickers.append(None)
                statuses.append("Total row")
                continue
                
            print(f"{idx + 1:2d}. {holding}")
            
            ticker, status = ticker_lookup.lookup_ticker(holding)
            tickers.append(ticker)
            statuses.append(status)
            
            if ticker:
                print(f"    ✅ {ticker} - {status}")
            else:
                print(f"    ⚪ No ticker - {status}")
            
            # Rate limiting for Yahoo Finance
            time.sleep(0.3)
        
        # Add ticker columns to DataFrame
        df['Ticker'] = tickers
        df['Ticker_Status'] = statuses
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Summary
        verified_tickers = sum(1 for t in tickers if t is not None)
        no_ticker = sum(1 for t in tickers if t is None)
        
        print(f"\n" + "=" * 60)
        print(f"✅ EXPORT COMPLETE")
        print(f"=" * 60)
        print(f"   File: {filename}")
        print(f"   Total records: {len(data)}")
        print(f"   Holdings with tickers: {verified_tickers}")
        print(f"   Holdings without tickers: {no_ticker}")
        print(f"   Columns: {list(df.columns)}")
        
        # Show ticker status distribution
        if statuses:
            print(f"\n🏷️  Ticker Status Summary:")
            status_counts = pd.Series(statuses).value_counts()
            for status, count in status_counts.items():
                print(f"   {status}: {count}")
        
        return filename

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            print("Browser closed.")


def get_browser_choice() -> str:
    """Get user's browser preference."""
    print("\nSelect browser:")
    print("1. Chrome (default)")
    print("2. Firefox")
    print("3. Safari")
    
    choice = input("Enter choice (1-3) or press Enter for Chrome: ").strip()
    
    browser_map = {"1": "chrome", "2": "firefox", "3": "safari", "": "chrome"}
    return browser_map.get(choice, "chrome")


def main() -> None:
    """Main execution function with integrated ticker lookup."""
    print("🏦 Empower Portfolio Holdings Scraper with Ticker Lookup")
    print("=" * 60)
    
    try:
        # Get browser choice
        browser = get_browser_choice()
        print(f"Using browser: {browser.title()}")
        
        # Initialize scraper
        scraper = EmpowerScraper(browser)
        
        # Setup browser
        print("\nSetting up browser...")
        scraper.setup_driver()
        
        # Login
        if not scraper.login():
            print("❌ Login failed. Exiting.")
            return
        
        # Navigate to holdings
        if not scraper.navigate_to_holdings():
            print("❌ Failed to navigate to holdings page. Exiting.")
            return
        
        # Scrape data
        holdings_data = scraper.scrape_holdings_data()
        
        if not holdings_data:
            print("❌ No data found. Please check the page structure.")
            return
        
        # Export to CSV
        csv_file = scraper.export_to_csv(holdings_data)
        print(f"\n🎉 Success! Data saved to: {csv_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        if 'scraper' in locals():
            scraper.cleanup()


if __name__ == "__main__":
    main()