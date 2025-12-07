# prediction_api.py
# FastAPI Application for GARCH Model Predictions

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import io

# Import your GARCHModel class and load function
# Make sure these are available - either copy them here or import from your notebook
from arch import arch_model

app = FastAPI(
    title="GARCH Volatility Prediction API",
    description="API for loading GARCH models and generating volatility predictions",
    version="2.0.0"
)

# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    model_path: str = Field(..., description="Path to saved GARCH model (.pkl file)")
    horizon: int = Field(default=5, ge=1, le=30, description="Forecast horizon (1-30 days)")
    
class RetrainRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    p: int = Field(default=1, ge=1, le=5, description="GARCH p parameter")
    q: int = Field(default=1, ge=1, le=5, description="GARCH q parameter")
    save_model: bool = Field(default=True, description="Save the trained model")

class CustomDataRequest(BaseModel):
    returns: List[float] = Field(..., description="List of return values")
    p: int = Field(default=1, ge=1, le=5)
    q: int = Field(default=1, ge=1, le=5)
    horizon: int = Field(default=5, ge=1, le=30)

# ============================================================================
# Helper Functions
# ============================================================================

def load_garch_model(filepath: str):
    """Load a saved GARCH model"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def forecast_from_model(model_data: dict, horizon: int = 5) -> Dict:
    """Generate forecast from loaded model"""
    fitted_model = model_data['fitted_model']
    predictions = fitted_model.forecast(horizon=horizon)
    
    variance_forecast = predictions.variance.values[-1]
    volatility_forecast = np.sqrt(variance_forecast)
    
    # Generate future dates
    last_date = fitted_model.data.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    
    predictions_dict = {
        date.strftime('%Y-%m-%d'): float(vol)
        for date, vol in zip(forecast_dates, volatility_forecast)
    }
    
    return predictions_dict

def train_garch_from_data(returns: pd.Series, p: int, q: int, horizon: int):
    """Train GARCH model from return data"""
    returns_scaled = returns.dropna() * 100
    
    model = arch_model(returns_scaled, vol='GARCH', p=p, q=q)
    fitted_model = model.fit(disp='off')
    
    predictions = fitted_model.forecast(horizon=horizon)
    variance_forecast = predictions.variance.values[-1]
    volatility_forecast = np.sqrt(variance_forecast)
    
    last_date = returns.index[-1] if isinstance(returns.index, pd.DatetimeIndex) else datetime.now()
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    
    predictions_dict = {
        date.strftime('%Y-%m-%d'): float(vol)
        for date, vol in zip(forecast_dates, volatility_forecast)
    }
    
    return fitted_model, predictions_dict

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GARCH Volatility Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "/predict": "POST - Load model and generate predictions",
            "/retrain": "POST - Retrain model with new data",
            "/predict/custom": "POST - Predict with custom return data",
            "/predict/upload": "POST - Upload CSV and predict",
            "/models/list": "GET - List available saved models",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_directory": os.path.exists('models')
    }

@app.post("/predict")
async def predict_volatility(request: PredictionRequest):
    """
    Load a saved GARCH model and generate volatility predictions
    
    Request body:
    {
        "model_path": "models/garch_model_AAPL.pkl",
        "horizon": 5
    }
    """
    try:
        # Load model
        model_data = load_garch_model(request.model_path)
        
        # Generate predictions
        predictions = forecast_from_model(model_data, request.horizon)
        
        # Get model info
        response = {
            "status": "success",
            "model_info": {
                "p": model_data['p'],
                "q": model_data['q'],
                "model_type": f"GARCH({model_data['p']},{model_data['q']})",
                "saved_date": model_data.get('save_date', 'Unknown')
            },
            "forecast_horizon": request.horizon,
            "predictions": predictions,
            "summary": {
                "mean_volatility": float(np.mean(list(predictions.values()))),
                "max_volatility": float(np.max(list(predictions.values()))),
                "min_volatility": float(np.min(list(predictions.values())))
            }
        }
        
        return JSONResponse(content=response)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/retrain")
async def retrain_model(request: RetrainRequest):
    """
    Fetch new data and retrain GARCH model
    
    Request body:
    {
        "ticker": "AAPL",
        "start_date": "2022-01-01",
        "end_date": "2024-01-01",
        "p": 1,
        "q": 1,
        "save_model": true
    }
    """
    try:
        import yfinance as yf
        
        # Fetch data
        start = request.start_date or (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end = request.end_date or datetime.now().strftime('%Y-%m-%d')
        
        stock = yf.Ticker(request.ticker)
        df = stock.history(start=start, end=end)
        
        if df.empty:
            raise ValueError(f"No data found for {request.ticker}")
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        returns = df['returns'].dropna()
        
        # Train model
        fitted_model, predictions = train_garch_from_data(
            returns, request.p, request.q, horizon=5
        )
        
        # Save model if requested
        model_path = None
        if request.save_model:
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'models/garch_{request.p}_{request.q}_{request.ticker}_{timestamp}.pkl'
            
            model_data = {
                'fitted_model': fitted_model,
                'p': request.p,
                'q': request.q,
                'returns': returns * 100,
                'predictions': None,
                'save_date': datetime.now().isoformat(),
                'ticker': request.ticker
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
        
        response = {
            "status": "success",
            "message": f"Model trained on {request.ticker}",
            "model_info": {
                "ticker": request.ticker,
                "p": request.p,
                "q": request.q,
                "data_points": len(returns),
                "date_range": {
                    "start": df.index[0].strftime('%Y-%m-%d'),
                    "end": df.index[-1].strftime('%Y-%m-%d')
                }
            },
            "model_path": model_path,
            "predictions": predictions,
            "model_summary": {
                "AIC": float(fitted_model.aic),
                "BIC": float(fitted_model.bic),
                "log_likelihood": float(fitted_model.loglikelihood)
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/predict/custom")
async def predict_custom_data(request: CustomDataRequest):
    """
    Train GARCH model on custom return data and generate predictions
    
    Request body:
    {
        "returns": [0.01, -0.02, 0.015, ...],
        "p": 1,
        "q": 1,
        "horizon": 5
    }
    """
    try:
        # Convert to pandas Series
        returns = pd.Series(request.returns)
        
        if len(returns) < 100:
            raise ValueError("Need at least 100 data points for reliable GARCH estimation")
        
        # Train model
        fitted_model, predictions = train_garch_from_data(
            returns, request.p, request.q, request.horizon
        )
        
        response = {
            "status": "success",
            "model_info": {
                "p": request.p,
                "q": request.q,
                "data_points": len(returns)
            },
            "predictions": predictions,
            "model_summary": {
                "AIC": float(fitted_model.aic),
                "BIC": float(fitted_model.bic)
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/upload")
async def predict_from_csv(
    file: UploadFile = File(...),
    p: int = 1,
    q: int = 1,
    horizon: int = 5
):
    """
    Upload a CSV file with returns data and generate predictions
    
    CSV should have a column named 'returns' or 'Returns'
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Find returns column
        returns_col = None
        for col in ['returns', 'Returns', 'return', 'Return']:
            if col in df.columns:
                returns_col = col
                break
        
        if returns_col is None:
            raise ValueError("CSV must contain a 'returns' column")
        
        returns = df[returns_col].dropna()
        
        if len(returns) < 100:
            raise ValueError("Need at least 100 data points")
        
        # Train model
        fitted_model, predictions = train_garch_from_data(
            returns, p, q, horizon
        )
        
        response = {
            "status": "success",
            "file_info": {
                "filename": file.filename,
                "data_points": len(returns)
            },
            "model_info": {
                "p": p,
                "q": q
            },
            "predictions": predictions,
            "model_summary": {
                "AIC": float(fitted_model.aic),
                "BIC": float(fitted_model.bic)
            }
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.get("/models/list")
async def list_models():
    """List all available saved models"""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        return {
            "status": "success",
            "models_directory": models_dir,
            "models": [],
            "count": 0
        }
    
    models = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(models_dir, filename)
            try:
                model_data = load_garch_model(filepath)
                models.append({
                    "filename": filename,
                    "path": filepath,
                    "p": model_data.get('p'),
                    "q": model_data.get('q'),
                    "ticker": model_data.get('ticker', 'Unknown'),
                    "saved_date": model_data.get('save_date', 'Unknown'),
                    "size_kb": round(os.path.getsize(filepath) / 1024, 2)
                })
            except:
                continue
    
    return {
        "status": "success",
        "models_directory": models_dir,
        "models": models,
        "count": len(models)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)