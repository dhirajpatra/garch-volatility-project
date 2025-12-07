# 📈 GARCH Volatility Modeling & Prediction API

A comprehensive end-to-end project for stock volatility analysis using GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models with a production-ready FastAPI application.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Project Overview

This project provides a complete pipeline for:
1. **Data Extraction** from Yahoo Finance API
2. **ETL Processing** with structured classes
3. **Volatility Analysis** and comparison
4. **GARCH Model Training** with parameter tuning
5. **Model Persistence** (save/load functionality)
6. **RESTful API** for predictions

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Step 1: Data Extraction API](#step-1-data-extraction-api)
  - [Step 2-4: GARCH Analysis](#step-2-4-garch-analysis)
  - [Step 5: Model Persistence](#step-5-model-persistence)
  - [Step 6: Prediction API](#step-6-prediction-api)
- [API Documentation](#api-documentation)
- [Model Theory](#model-theory)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality
- ✅ **Automated Data Pipeline**: ETL architecture with Extract, Transform, Load classes
- ✅ **SQLite Database**: Persistent storage for historical stock data
- ✅ **GARCH Modeling**: Support for GARCH(p,q) models with configurable parameters
- ✅ **Parameter Tuning**: Automatic selection of optimal p and q values using AIC/BIC
- ✅ **Model Persistence**: Save and load trained models for reuse
- ✅ **Comprehensive Visualization**: Volatility plots, ACF/PACF, residual diagnostics

### API Features
- ✅ **RESTful Endpoints**: FastAPI-based prediction service
- ✅ **Multiple Input Methods**: Pre-trained models, fresh training, custom data, CSV upload
- ✅ **Interactive Documentation**: Auto-generated Swagger UI
- ✅ **Error Handling**: Robust defensive programming
- ✅ **Model Management**: List, load, and manage saved models

## 📁 Project Structure

```
garch-volatility-project/
│
├── app.py                      # Step 1: Data extraction FastAPI
├── prediction_api.py           # Step 6: Prediction FastAPI
├── garch_analysis.ipynb        # Steps 2-5: Main analysis notebook
│
├── models/                     # Saved GARCH models
│   ├── garch_1_1_AAPL_*.pkl
│   ├── garch_1_1_GOOGL_*.pkl
│   └── ...
│
├── data/
│   └── stock_data.db          # SQLite database
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/garch-volatility-project.git
cd garch-volatility-project
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements.txt
```text
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
yfinance==0.2.32
pandas==2.1.3
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
statsmodels==0.14.0
arch==6.2.0
scipy==1.11.4
python-multipart==0.0.6
```

## 🎬 Quick Start

### 1. Start Data Extraction API (Port 8000)
```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

### 2. Start Prediction API (Port 8001)
```bash
python prediction_api.py
# or
uvicorn prediction_api:app --reload --port 8001
```

### 3. Run Jupyter Notebook
```bash
jupyter notebook garch_analysis.ipynb
```

### 4. Quick Test
```bash
# Test data extraction
curl "http://localhost:8000/stock/AAPL"

# Test prediction API
curl "http://localhost:8001/health"
```

## 📚 Detailed Usage

### Step 1: Data Extraction API

**Purpose**: Fetch stock data from Yahoo Finance with defensive error handling

**Endpoints**:

#### GET `/stock/{ticker}`
Fetch stock data for a specific ticker.

```bash
curl "http://localhost:8000/stock/AAPL?start_date=2023-01-01&end_date=2024-01-01"
```

**Response**:
```json
{
  "ticker": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "data_points": 252,
  "summary_statistics": {
    "mean_return": 0.0012,
    "std_return": 0.0145,
    "min_price": 124.17,
    "max_price": 199.62,
    "current_price": 185.92
  },
  "data": [...]
}
```

#### GET `/stock/{ticker}/volatility`
Get rolling volatility calculation.

```bash
curl "http://localhost:8000/stock/AAPL/volatility?window=30"
```

### Step 2-4: GARCH Analysis

**Run in Jupyter Notebook**: `garch_analysis.ipynb`

#### Step 2: ETL Pipeline

**Extract, Transform, Load** stock data into SQLite database.

```python
# Run ETL for multiple stocks
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

for ticker in tickers:
    run_etl_pipeline(ticker, start_date='2022-01-01')
```

**Output**:
- Raw data fetched from Yahoo Finance
- Returns calculated (simple, log, squared)
- Rolling volatility computed
- Data stored in `stock_data.db`

#### Step 3: Stock Comparison

**Compare volatility** across multiple stocks.

```python
# Load stocks from database
stocks_data = load_multiple_stocks(['AAPL', 'GOOGL', 'MSFT'])

# Generate summary statistics
print(summary_statistics(stocks_data))

# Visualize comparisons
plot_volatility_comparison(stocks_data)
plot_acf_comparison(stocks_data)
```

**Visualizations**:
1. **Rolling Volatility Time Series**: Compare 30-day volatility
2. **Squared Returns**: Volatility proxy over time
3. **ACF Plots**: Autocorrelation of squared returns
4. **Summary Statistics Table**: Mean, std, skewness, kurtosis

#### Step 4: GARCH Modeling

**Build and train** GARCH models with parameter tuning.

```python
# Select stock
ticker = 'AAPL'
returns = stocks_data[ticker]['returns'].dropna()

# Tune parameters (find optimal p, q)
param_results = tune_garch_parameters(returns, max_p=3, max_q=3)
print(param_results)

# Build GARCH model
garch = GARCHModel(returns, p=1, q=1)
garch.build_model()
garch.fit()

# Visualize results
garch.plot_volatility()
garch.plot_residuals_diagnostics()

# Generate forecast
garch.forecast(horizon=5)
predictions = garch.get_predictions_dict()
print(predictions)
```

**Model Output**:
```
                     Constant Mean - GARCH Model Results                      
==============================================================================
Dep. Variable:                returns   R-squared:                       0.000
Mean Model:             Constant Mean   Adj. R-squared:                  0.000
Vol Model:                      GARCH   Log-Likelihood:               -123.456
Distribution:                  Normal   AIC:                           252.912
                                        BIC:                           268.234
                               No. Observations:                  500
==============================================================================
                 coef    std err          t      P>|t|    95.0% Conf. Int.
------------------------------------------------------------------------------
mu             0.0567  1.234e-02      4.595  4.319e-06 [3.253e-02,8.087e-02]
omega          0.0123  4.567e-03      2.694  7.066e-03 [3.289e-03,2.131e-02]
alpha[1]       0.0890  2.345e-02      3.796  1.472e-04 [4.303e-02,1.350e-01]
beta[1]        0.9012  1.234e-02     73.051      0.000 [    0.877,    0.925]
==============================================================================
```

### Step 5: Model Persistence

**Save and load** trained models for reuse.

```python
# Save model
saved_path = garch.save_model(model_name=f'garch_model_{ticker}')
# Output: ✓ Model saved successfully to: models/garch_model_AAPL.pkl

# List all saved models
list_saved_models()
# Output:
# Found 3 saved model(s):
#   1. garch_1_1_AAPL_20241207_143022.pkl
#   2. garch_1_1_GOOGL_20241207_143045.pkl
#   3. garch_2_2_MSFT_20241207_143110.pkl

# Load model later
loaded_garch = load_garch_model('models/garch_model_AAPL.pkl')
loaded_garch.forecast(horizon=5)
```

### Step 6: Prediction API

**Deploy models** via RESTful API for production use.

#### Endpoint 1: Predict with Saved Model

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "models/garch_1_1_AAPL_20241207_143022.pkl",
    "horizon": 5
  }'
```

**Response**:
```json
{
  "status": "success",
  "model_info": {
    "p": 1,
    "q": 1,
    "model_type": "GARCH(1,1)",
    "saved_date": "2024-12-07T14:30:22"
  },
  "forecast_horizon": 5,
  "predictions": {
    "2024-12-08": 1.2345,
    "2024-12-09": 1.2567,
    "2024-12-10": 1.2789,
    "2024-12-11": 1.2901,
    "2024-12-12": 1.3023
  },
  "summary": {
    "mean_volatility": 1.2725,
    "max_volatility": 1.3023,
    "min_volatility": 1.2345
  }
}
```

#### Endpoint 2: Retrain Model

```bash
curl -X POST "http://localhost:8001/retrain" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "TSLA",
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "p": 1,
    "q": 1,
    "save_model": true
  }'
```

#### Endpoint 3: Predict with Custom Data

```bash
curl -X POST "http://localhost:8001/predict/custom" \
  -H "Content-Type: application/json" \
  -d '{
    "returns": [0.01, -0.02, 0.015, 0.008, -0.01, 0.012, ...],
    "p": 1,
    "q": 1,
    "horizon": 5
  }'
```

#### Endpoint 4: Upload CSV

```bash
curl -X POST "http://localhost:8001/predict/upload" \
  -F "file=@returns_data.csv" \
  -F "p=1" \
  -F "q=1" \
  -F "horizon=5"
```

**CSV Format**:
```csv
date,returns
2024-01-01,0.012
2024-01-02,-0.008
2024-01-03,0.015
...
```

#### Endpoint 5: List Models

```bash
curl "http://localhost:8001/models/list"
```

**Response**:
```json
{
  "status": "success",
  "models_directory": "models",
  "models": [
    {
      "filename": "garch_1_1_AAPL_20241207_143022.pkl",
      "path": "models/garch_1_1_AAPL_20241207_143022.pkl",
      "p": 1,
      "q": 1,
      "ticker": "AAPL",
      "saved_date": "2024-12-07T14:30:22",
      "size_kb": 245.67
    }
  ],
  "count": 1
}
```

## 📖 API Documentation

### Interactive Documentation

Once the APIs are running, access interactive documentation:

- **Data Extraction API**: http://localhost:8000/docs
- **Prediction API**: http://localhost:8001/docs

### Authentication

Currently, the APIs are **open** without authentication. For production:

1. Add API key authentication
2. Implement rate limiting
3. Use HTTPS/TLS
4. Add user management

Example with API key:
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
```

## 🧮 Model Theory

### What is GARCH?

**GARCH (Generalized Autoregressive Conditional Heteroskedasticity)** models are used to estimate the volatility of financial returns.

#### Mathematical Formulation

**GARCH(p,q) Model**:

1. **Return equation**:
   ```
   r_t = μ + ε_t
   ε_t = σ_t * z_t
   ```
   where z_t ~ N(0,1)

2. **Volatility equation**:
   ```
   σ²_t = ω + Σ(α_i * ε²_{t-i}) + Σ(β_j * σ²_{t-j})
            i=1 to q              j=1 to p
   ```

#### Parameters

- **p**: Order of GARCH term (lagged variance)
- **q**: Order of ARCH term (lagged squared residuals)
- **ω (omega)**: Constant term
- **α (alpha)**: ARCH coefficients
- **β (beta)**: GARCH coefficients

#### Common Models

- **GARCH(1,1)**: Most popular, captures volatility clustering
- **GARCH(2,1)**: More complex short-term dynamics
- **GARCH(1,2)**: More complex long-term persistence

### Why GARCH?

1. **Volatility Clustering**: High volatility periods followed by high volatility
2. **Fat Tails**: Better captures extreme events than normal distribution
3. **Mean Reversion**: Volatility tends to revert to long-term average
4. **Leverage Effects**: Can be extended (EGARCH, TGARCH) for asymmetric responses

### Model Selection

Use **AIC (Akaike Information Criterion)** or **BIC (Bayesian Information Criterion)**:
- Lower values indicate better fit
- BIC penalizes complexity more than AIC
- Balance between fit and parsimony

## 💡 Examples

### Example 1: Complete Workflow

```python
# 1. Extract and store data
run_etl_pipeline('AAPL', start_date='2022-01-01')

# 2. Load and analyze
loader = Load()
loader.connect()
df = loader.load_data('AAPL')
df.set_index('date', inplace=True)

# 3. Train GARCH model
returns = df['returns'].dropna()
garch = GARCHModel(returns, p=1, q=1)
garch.build_model()
garch.fit()

# 4. Generate predictions
garch.forecast(horizon=10)
predictions = garch.get_predictions_dict()

# 5. Save model
garch.save_model(model_name='my_garch_model')

# 6. Use via API
# Start API and call: POST /predict with saved model
```

### Example 2: Compare Multiple Stocks

```python
# ETL for multiple stocks
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
for ticker in tickers:
    run_etl_pipeline(ticker)

# Load and compare
stocks_data = load_multiple_stocks(tickers)

# Statistics
stats = summary_statistics(stocks_data)
print(stats.sort_values('Annualized Vol', ascending=False))

# Visualize
plot_volatility_comparison(stocks_data)
plot_acf_comparison(stocks_data)

# Train best model for most volatile
most_volatile = stats.loc[stats['Annualized Vol'].idxmax(), 'Ticker']
returns = stocks_data[most_volatile]['returns']
garch = GARCHModel(returns, p=1, q=1)
garch.fit()
```

### Example 3: Parameter Tuning

```python
# Test multiple configurations
results = tune_garch_parameters(returns, max_p=3, max_q=3)
print(results.head())

# Select best model
best_p = int(results.iloc[0]['p'])
best_q = int(results.iloc[0]['q'])

# Train with best parameters
garch_best = GARCHModel(returns, p=best_p, q=best_q)
garch_best.fit()
garch_best.plot_volatility()
```

### Example 4: Production Pipeline

```python
import schedule
import time

def daily_update():
    """Daily model update job"""
    # Fetch latest data
    run_etl_pipeline('AAPL')
    
    # Retrain model
    loader = Load()
    loader.connect()
    df = loader.load_data('AAPL')
    returns = df['returns'].dropna()
    
    garch = GARCHModel(returns, p=1, q=1)
    garch.fit()
    
    # Save updated model
    garch.save_model(model_name='garch_model_AAPL_latest')
    
    print(f"Model updated at {datetime.now()}")

# Schedule daily at 6 PM
schedule.every().day.at("18:00").do(daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "No module named 'arch'"
```bash
pip install arch
```

#### 2. "Model file not found"
Check if `models/` directory exists:
```python
import os
os.makedirs('models', exist_ok=True)
```

#### 3. "Convergence warning"
- Increase max iterations in model.fit()
- Try different starting values
- Check data quality (remove outliers)

```python
fitted_model = model.fit(disp='off', options={'maxiter': 1000})
```

#### 4. "Not enough data points"
GARCH models need sufficient data:
- Minimum: 100 observations
- Recommended: 250+ observations (1 year daily data)

#### 5. API Connection Refused
Check if port is already in use:
```bash
# Find process using port 8001
lsof -i :8001

# Kill process
kill -9 <PID>

# Use different port
uvicorn prediction_api:app --port 8002
```

### Data Quality Checks

```python
def validate_returns(returns):
    """Validate return data quality"""
    issues = []
    
    if len(returns) < 100:
        issues.append(f"Too few observations: {len(returns)}")
    
    if returns.isna().sum() > 0:
        issues.append(f"Missing values: {returns.isna().sum()}")
    
    if (returns.abs() > 0.5).any():
        issues.append("Extreme returns detected (>50%)")
    
    if returns.std() == 0:
        issues.append("Zero variance")
    
    return issues if issues else ["Data OK"]

# Use before modeling
issues = validate_returns(returns)
print(issues)
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/garch-volatility-project.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance API** for financial data
- **arch** library by Kevin Sheppard
- **FastAPI** framework
- **Statsmodels** for time series analysis

## 📞 Contact

- **Author**: Dhiraj Patra
- **Email**: dhiraj.patra@gmail.com
- **GitHub**: [@dhirajpatra](https://github.com/dhirajpatra)
- **LinkedIn**: [Dhiraj Patra](https://linkedin.com/in/dhirajpatra)

## 🗺️ Roadmap

### Future Enhancements

- [ ] Add EGARCH and TGARCH models
- [ ] Implement multivariate GARCH (DCC, BEKK)
- [ ] Add real-time streaming data
- [ ] Create web dashboard with Streamlit
- [ ] Add portfolio optimization features
- [ ] Implement backtesting framework
- [ ] Add machine learning hybrid models
- [ ] Support for crypto and forex data
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)

## 📊 Performance Benchmarks

Typical performance on standard hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Fetch 1 year data | ~2s | Yahoo Finance API |
| GARCH(1,1) fitting | ~0.5s | 250 data points |
| Parameter tuning (3x3) | ~5s | 9 model combinations |
| API prediction request | ~0.1s | Pre-trained model |
| ETL pipeline (1 stock) | ~3s | Full workflow |

## 📚 References

1. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity"
2. Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity"
3. Hansen, P. R., & Lunde, A. (2005). "A forecast comparison of volatility models"
4. Kevin Sheppard's arch library documentation

---

⭐ **Star this repo** if you find it helpful!

📖 **Found a bug?** Open an issue!

💬 **Have questions?** Start a discussion!