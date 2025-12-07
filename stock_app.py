from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uvicorn

app = FastAPI(
    title="Stock Volatility API",
    description="API for fetching stock data and GARCH volatility predictions",
    version="1.0.0"
)

class StockRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class StockDataExtractor:
    """Class to extract stock data from Yahoo Finance API"""
    
    @staticmethod
    def fetch_stock_data(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with stock data
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            return df
        
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Volatility API",
        "version": "1.0.0",
        "endpoints": {
            "/stock/{ticker}": "Get stock data",
            "/stock/data": "Post request for stock data with date range",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/stock/{ticker}")
async def get_stock_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    GET endpoint to fetch stock data
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, GOOGL, WBD)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
    """
    try:
        extractor = StockDataExtractor()
        df = extractor.fetch_stock_data(ticker, start_date, end_date)
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Convert to dictionary format
        result = {
            "ticker": ticker.upper(),
            "start_date": df.index[0].strftime('%Y-%m-%d'),
            "end_date": df.index[-1].strftime('%Y-%m-%d'),
            "data_points": len(df),
            "summary_statistics": {
                "mean_return": float(df['returns'].mean()),
                "std_return": float(df['returns'].std()),
                "min_price": float(df['Close'].min()),
                "max_price": float(df['Close'].max()),
                "current_price": float(df['Close'].iloc[-1])
            },
            "data": df.reset_index().to_dict(orient='records')
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stock/data")
async def post_stock_data(request: StockRequest):
    """
    POST endpoint to fetch stock data with request body
    
    Body:
        {
            "ticker": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01"
        }
    """
    try:
        extractor = StockDataExtractor()
        df = extractor.fetch_stock_data(
            request.ticker, 
            request.start_date, 
            request.end_date
        )
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        result = {
            "ticker": request.ticker.upper(),
            "start_date": df.index[0].strftime('%Y-%m-%d'),
            "end_date": df.index[-1].strftime('%Y-%m-%d'),
            "data_points": len(df),
            "summary_statistics": {
                "mean_return": float(df['returns'].mean()),
                "std_return": float(df['returns'].std()),
                "min_price": float(df['Close'].min()),
                "max_price": float(df['Close'].max()),
                "current_price": float(df['Close'].iloc[-1])
            },
            "data": df.reset_index().to_dict(orient='records')
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stock/{ticker}/volatility")
async def get_stock_volatility(
    ticker: str,
    window: int = 30,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get rolling volatility for a stock
    
    Args:
        ticker: Stock ticker symbol
        window: Rolling window size for volatility calculation
        start_date: Optional start date
        end_date: Optional end date
    """
    try:
        extractor = StockDataExtractor()
        df = extractor.fetch_stock_data(ticker, start_date, end_date)
        
        # Calculate returns and rolling volatility
        df['returns'] = df['Close'].pct_change()
        df['rolling_volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        result = {
            "ticker": ticker.upper(),
            "window": window,
            "start_date": df.index[0].strftime('%Y-%m-%d'),
            "end_date": df.index[-1].strftime('%Y-%m-%d'),
            "annualized_volatility": float(df['returns'].std() * np.sqrt(252)),
            "volatility_data": [
                {
                    "date": idx.strftime('%Y-%m-%d'),
                    "volatility": float(vol) if pd.notna(vol) else None
                }
                for idx, vol in df['rolling_volatility'].items()
            ]
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Run with: python app.py
    # Or: uvicorn app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)