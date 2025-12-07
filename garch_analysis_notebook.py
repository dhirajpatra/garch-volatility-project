# Stock Volatility Analysis with GARCH Model
# Steps 2-4: ETL, Comparison, and GARCH Modeling

import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 2: ETL (Extract, Transform, Load) with Classes
# ============================================================================

class Extract:
    """Extract stock data from Yahoo Finance API"""
    
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None):
        self.ticker = ticker.upper()
        self.start_date = start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            print(f"Fetching data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            df = stock.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            print(f"✓ Successfully fetched {len(df)} records")
            return df
        
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            return pd.DataFrame()


class Transform:
    """Transform raw stock data into analysis-ready format"""
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics"""
        df = df.copy()
        
        # Simple returns
        df['returns'] = df['Close'].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Squared returns (proxy for volatility)
        df['returns_squared'] = df['returns'] ** 2
        
        # Rolling volatility (30-day window)
        df['rolling_vol_30d'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
        
        return df
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        df = df.copy()
        
        # Remove NaN values
        df = df.dropna()
        
        # Reset index to make date a column
        df = df.reset_index()
        df.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Data cleaned: {len(df)} valid records")
        return df


class Load:
    """Load data into SQLite database"""
    
    def __init__(self, db_path: str = 'stock_data.db'):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)
        print(f"✓ Connected to database: {self.db_path}")
    
    def create_table(self):
        """Create stock data table if it doesn't exist"""
        create_query = """
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            returns REAL,
            log_returns REAL,
            returns_squared REAL,
            rolling_vol_30d REAL,
            UNIQUE(ticker, date)
        )
        """
        self.conn.execute(create_query)
        self.conn.commit()
        print("✓ Table created/verified")
    
    def save_data(self, df: pd.DataFrame, ticker: str):
        """Save data to database"""
        df_to_save = df.copy()
        df_to_save['ticker'] = ticker
        
        # Select relevant columns
        columns = ['ticker', 'date', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'returns', 'log_returns', 'returns_squared', 'rolling_vol_30d']
        
        # Rename columns to lowercase
        df_to_save.columns = [col.lower() for col in df_to_save.columns]
        
        try:
            df_to_save[columns].to_sql('stock_data', self.conn, 
                                       if_exists='append', index=False)
            print(f"✓ Data saved for {ticker}: {len(df_to_save)} records")
        except sqlite3.IntegrityError:
            # Handle duplicates by replacing
            print(f"⚠ Duplicate data detected. Updating existing records...")
            self.conn.execute(f"DELETE FROM stock_data WHERE ticker = '{ticker}'")
            df_to_save[columns].to_sql('stock_data', self.conn, 
                                       if_exists='append', index=False)
            print(f"✓ Data updated for {ticker}")
    
    def load_data(self, ticker: str = None) -> pd.DataFrame:
        """Load data from database"""
        if ticker:
            query = f"SELECT * FROM stock_data WHERE ticker = '{ticker}' ORDER BY date"
        else:
            query = "SELECT * FROM stock_data ORDER BY ticker, date"
        
        df = pd.read_sql_query(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} records")
        return df
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")


# ============================================================================
# ETL Pipeline Function
# ============================================================================

def run_etl_pipeline(ticker: str, start_date: str = None, end_date: str = None):
    """Run complete ETL pipeline for a stock"""
    print(f"\n{'='*60}")
    print(f"Running ETL Pipeline for {ticker}")
    print(f"{'='*60}\n")
    
    # Extract
    extractor = Extract(ticker, start_date, end_date)
    raw_data = extractor.fetch_data()
    
    if raw_data.empty:
        print("Pipeline failed: No data extracted")
        return None
    
    # Transform
    transformer = Transform()
    transformed_data = transformer.calculate_returns(raw_data)
    clean_data = transformer.clean_data(transformed_data)
    
    # Load
    loader = Load()
    loader.connect()
    loader.create_table()
    loader.save_data(clean_data, ticker)
    loader.close()
    
    print(f"\n{'='*60}")
    print(f"ETL Pipeline Complete for {ticker}")
    print(f"{'='*60}\n")
    
    return clean_data


# ============================================================================
# STEP 3: Comparing Stock Data
# ============================================================================

def load_multiple_stocks(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Load multiple stocks from database"""
    loader = Load()
    loader.connect()
    
    stocks_data = {}
    for ticker in tickers:
        df = loader.load_data(ticker)
        if not df.empty:
            df.set_index('date', inplace=True)
            stocks_data[ticker] = df
    
    loader.close()
    return stocks_data


def plot_volatility_comparison(stocks_data: Dict[str, pd.DataFrame], figsize=(14, 8)):
    """Plot volatility time series for multiple stocks"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Rolling Volatility
    ax1 = axes[0]
    for ticker, df in stocks_data.items():
        ax1.plot(df.index, df['rolling_vol_30d'], label=ticker, linewidth=2)
    
    ax1.set_title('30-Day Rolling Volatility Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Annualized Volatility')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Squared Returns (Volatility Proxy)
    ax2 = axes[1]
    for ticker, df in stocks_data.items():
        ax2.plot(df.index, df['returns_squared'], label=ticker, alpha=0.7, linewidth=1)
    
    ax2.set_title('Squared Returns (Volatility Proxy)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Squared Returns')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_acf_comparison(stocks_data: Dict[str, pd.DataFrame], lags=40, figsize=(14, 10)):
    """Plot ACF of squared returns for multiple stocks"""
    n_stocks = len(stocks_data)
    fig, axes = plt.subplots(n_stocks, 1, figsize=figsize)
    
    if n_stocks == 1:
        axes = [axes]
    
    for idx, (ticker, df) in enumerate(stocks_data.items()):
        returns_sq = df['returns_squared'].dropna()
        plot_acf(returns_sq, lags=lags, ax=axes[idx])
        axes[idx].set_title(f'ACF of Squared Returns - {ticker}', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def summary_statistics(stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate summary statistics for multiple stocks"""
    stats = []
    
    for ticker, df in stocks_data.items():
        returns = df['returns'].dropna()
        stats.append({
            'Ticker': ticker,
            'Mean Return': returns.mean(),
            'Std Return': returns.std(),
            'Annualized Vol': returns.std() * np.sqrt(252),
            'Min Return': returns.min(),
            'Max Return': returns.max(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis()
        })
    
    return pd.DataFrame(stats)


# ============================================================================
# STEP 4: Building GARCH Model
# ============================================================================

class GARCHModel:
    """GARCH Model for volatility forecasting"""
    
    def __init__(self, returns: pd.Series, p: int = 1, q: int = 1):
        """
        Initialize GARCH model
        
        Args:
            returns: Time series of returns
            p: GARCH order (lag of squared residuals)
            q: ARCH order (lag of conditional variance)
        """
        self.returns = returns.dropna() * 100  # Scale to percentage
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.predictions = None
    
    def build_model(self, vol: str = 'GARCH', dist: str = 'normal'):
        """Build GARCH model"""
        self.model = arch_model(
            self.returns,
            vol=vol,
            p=self.p,
            q=self.q,
            dist=dist
        )
        print(f"✓ GARCH({self.p},{self.q}) model built with {dist} distribution")
    
    def fit(self):
        """Fit the model to data"""
        print("Fitting GARCH model...")
        self.fitted_model = self.model.fit(disp='off')
        print("✓ Model fitted successfully\n")
        print(self.fitted_model.summary())
        return self.fitted_model
    
    def forecast(self, horizon: int = 5):
        """Generate volatility forecast"""
        self.predictions = self.fitted_model.forecast(horizon=horizon)
        return self.predictions
    
    def plot_volatility(self, figsize=(14, 8)):
        """Plot conditional volatility"""
        if self.fitted_model is None:
            print("Error: Model not fitted yet")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Conditional Volatility
        cond_vol = self.fitted_model.conditional_volatility
        ax1 = axes[0]
        ax1.plot(cond_vol.index, cond_vol.values, linewidth=2, color='#e74c3c')
        ax1.set_title('GARCH Conditional Volatility', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Volatility (%)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Standardized Residuals
        std_resid = self.fitted_model.std_resid
        ax2 = axes[1]
        ax2.plot(std_resid.index, std_resid.values, linewidth=1, alpha=0.7, color='#3498db')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Standardized Residuals', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Std Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals_diagnostics(self, lags=40, figsize=(14, 10)):
        """Plot residual diagnostics"""
        if self.fitted_model is None:
            print("Error: Model not fitted yet")
            return
        
        std_resid = self.fitted_model.std_resid
        std_resid_sq = std_resid ** 2
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # ACF of Standardized Residuals
        plot_acf(std_resid.dropna(), lags=lags, ax=axes[0, 0])
        axes[0, 0].set_title('ACF of Standardized Residuals', fontweight='bold')
        
        # PACF of Standardized Residuals
        plot_pacf(std_resid.dropna(), lags=lags, ax=axes[0, 1])
        axes[0, 1].set_title('PACF of Standardized Residuals', fontweight='bold')
        
        # ACF of Squared Standardized Residuals
        plot_acf(std_resid_sq.dropna(), lags=lags, ax=axes[1, 0])
        axes[1, 0].set_title('ACF of Squared Std Residuals', fontweight='bold')
        
        # Histogram of Standardized Residuals
        axes[1, 1].hist(std_resid.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Distribution of Std Residuals', fontweight='bold')
        axes[1, 1].set_xlabel('Standardized Residuals')
        axes[1, 1].set_ylabel('Frequency')
        
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_predictions_dict(self, forecast_dates: pd.DatetimeIndex = None) -> Dict:
        """
        Convert predictions to dictionary with ISO dates
        
        Args:
            forecast_dates: Future dates for forecast (if None, uses last dates)
        
        Returns:
            Dictionary with dates and predicted volatility
        """
        if self.predictions is None:
            print("Error: No predictions available. Run forecast() first.")
            return {}
        
        variance_forecast = self.predictions.variance.values[-1]
        volatility_forecast = np.sqrt(variance_forecast)
        
        if forecast_dates is None:
            last_date = self.returns.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=len(volatility_forecast),
                freq='D'
            )
        
        predictions_dict = {
            date.strftime('%Y-%m-%d'): float(vol)
            for date, vol in zip(forecast_dates, volatility_forecast)
        }
        
        return predictions_dict

    def save_model(self, filepath: str = None, model_name: str = None):
        """
        Save the fitted GARCH model to disk
        
        Args:
            filepath: Directory path to save model (default: 'models/')
            model_name: Name for the model file (default: auto-generated)
        
        Returns:
            Full path of saved model
        """
        if self.fitted_model is None:
            print("Error: No fitted model to save. Fit the model first.")
            return None
        
        # Create models directory if it doesn't exist
        if filepath is None:
            filepath = 'models'
        
        os.makedirs(filepath, exist_ok=True)
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f'garch_{self.p}_{self.q}_{timestamp}.pkl'
        
        # Ensure .pkl extension
        if not model_name.endswith('.pkl'):
            model_name += '.pkl'
        
        full_path = os.path.join(filepath, model_name)
        
        # Save model and metadata
        model_data = {
            'fitted_model': self.fitted_model,
            'p': self.p,
            'q': self.q,
            'returns': self.returns,
            'predictions': self.predictions,
            'save_date': datetime.now().isoformat()
        }
        
        with open(full_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved successfully to: {full_path}")
        return full_path
    

# Standalone function to load saved models
# (Add this after the GARCHModel class definition, outside the class)
def load_garch_model(filepath: str) -> GARCHModel:
    """
    Load a saved GARCH model from disk
    
    Args:
        filepath: Path to the saved model file (.pkl)
    
    Returns:
        GARCHModel object with fitted model loaded
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    print(f"Loading model from: {filepath}")
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Recreate GARCHModel object
    garch = GARCHModel(
        returns=model_data['returns'] / 100,  # Unscale returns
        p=model_data['p'],
        q=model_data['q']
    )
    
    # Restore fitted model and predictions
    garch.fitted_model = model_data['fitted_model']
    garch.predictions = model_data['predictions']
    
    # Build the model structure
    garch.build_model()
    
    print(f"✓ Model loaded: GARCH({model_data['p']},{model_data['q']})")
    print(f"  Saved on: {model_data['save_date']}")
    
    return garch


def list_saved_models(directory: str = 'models') -> list:
    """
    List all saved GARCH models in a directory
    
    Args:
        directory: Directory to search for models
    
    Returns:
        List of model filenames
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist")
        return []
    
    models = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    if models:
        print(f"Found {len(models)} saved model(s):")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    else:
        print(f"No saved models found in '{directory}'")
    
    return models

def tune_garch_parameters(returns: pd.Series, max_p: int = 3, max_q: int = 3):
    """
    Tune GARCH(p,q) parameters using AIC/BIC
    
    Args:
        returns: Return series
        max_p: Maximum p value to test
        max_q: Maximum q value to test
    
    Returns:
        DataFrame with model comparison
    """
    results = []
    returns_scaled = returns.dropna() * 100
    
    print("Tuning GARCH parameters...\n")
    
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                model = arch_model(returns_scaled, vol='GARCH', p=p, q=q)
                fitted = model.fit(disp='off')
                
                results.append({
                    'p': p,
                    'q': q,
                    'AIC': fitted.aic,
                    'BIC': fitted.bic,
                    'Log-Likelihood': fitted.loglikelihood
                })
                print(f"✓ GARCH({p},{q}) - AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
            
            except Exception as e:
                print(f"✗ GARCH({p},{q}) failed: {e}")
    
    results_df = pd.DataFrame(results).sort_values('AIC')
    print(f"\nBest model by AIC: GARCH({results_df.iloc[0]['p']:.0f},{results_df.iloc[0]['q']:.0f})")
    
    return results_df


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    
    # STEP 2: Run ETL Pipeline
    print("\n" + "="*70)
    print("STEP 2: ETL PIPELINE")
    print("="*70)
    
    # Extract, Transform, Load stock data
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    for ticker in tickers:
        run_etl_pipeline(ticker, start_date='2022-01-01')
    
    # STEP 3: Compare Stock Data
    print("\n" + "="*70)
    print("STEP 3: COMPARING STOCK DATA")
    print("="*70)
    
    # Load all stocks
    stocks_data = load_multiple_stocks(tickers)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(summary_statistics(stocks_data))
    
    # Plot comparisons
    plot_volatility_comparison(stocks_data)
    plot_acf_comparison(stocks_data)
    
    # STEP 4: Build GARCH Model
    print("\n" + "="*70)
    print("STEP 4: BUILDING GARCH MODEL")
    print("="*70)
    
    # Select one stock for GARCH modeling
    ticker = 'AAPL'
    returns = stocks_data[ticker]['returns'].dropna()
    
    # Tune parameters (optional)
    print(f"\nTuning GARCH parameters for {ticker}...")
    param_results = tune_garch_parameters(returns, max_p=2, max_q=2)
    print("\nParameter Tuning Results:")
    print(param_results)
    
    # Build and fit GARCH model
    print(f"\nBuilding GARCH model for {ticker}...")
    garch = GARCHModel(returns, p=1, q=1)
    garch.build_model()
    garch.fit()
    
    # Visualize results
    garch.plot_volatility()
    garch.plot_residuals_diagnostics()
    
    # Generate forecast
    print("\nGenerating 5-day volatility forecast...")
    garch.forecast(horizon=5)
    predictions_dict = garch.get_predictions_dict()
    print("\nForecasted Volatility:")
    for date, vol in predictions_dict.items():
        print(f"  {date}: {vol:.4f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    # STEP 5: Save and Reuse Model
    print("\n" + "="*70)
    print("STEP 5: SAVING AND LOADING MODELS")
    print("="*70)
    
    # Save the fitted model
    print("\nSaving the GARCH model...")
    saved_path = garch.save_model(model_name=f'garch_model_{ticker}')
    
    # List all saved models
    print("\n" + "-"*70)
    list_saved_models()
    
    # Load the model back
    print("\n" + "-"*70)
    print("Testing model loading...")
    loaded_garch = load_garch_model(saved_path)
    
    # Verify it works by making predictions
    print("\nMaking predictions with loaded model...")
    loaded_garch.forecast(horizon=5)
    loaded_predictions = loaded_garch.get_predictions_dict()
    print("\nLoaded Model Predictions:")
    for date, vol in loaded_predictions.items():
        print(f"  {date}: {vol:.4f}%")
    
    print("\n" + "="*70)
    print("MODEL SAVE/LOAD SUCCESSFUL")
    print("="*70)