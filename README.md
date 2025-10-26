# 📈 Sentiment-Driven Volatility Forecaster (SDVF) – AI Trading Terminal  


## 🧠 Project Overview

**Sentiment-Driven Volatility Forecaster (SDVF)** is a sophisticated AI-powered tool for forecasting stock market volatility and price direction using a **2-Stage Autoregressive Distributed Lag (ARDL)** model.  
It enhances traditional econometric forecasting by integrating **financial news sentiment analysis** using transformer-based models from **Hugging Face**.

The project provides:
- A **Command-Line Interface (CLI)** for quick terminal-based forecasting.
- An **interactive AI Trading Terminal** built with **Streamlit**, enabling real-time data visualization, sentiment summaries, and trading recommendations.

This tool offers **quantitative insights** into potential market movements by combining:
- Historical price data  
- Real-time financial news sentiment  

---

## ⚙️ Core Methodology & Data Pipeline

### **1. Data Acquisition**
- **Stock Prices:**  
  Fetches up to **10 years** of daily OHLCV (Open, High, Low, Close, Volume) data using the `yfinance` library.
- **Financial News:**  
  Retrieves **recent headlines and article summaries** related to the ticker using the **NewsAPI**.

---

### **2. AI Sentiment Analysis & Integration**
- **Real-time Sentiment:**  
  Uses **FinBERT** (a transformer model fine-tuned for financial sentiment) to compute a sentiment score between **-1 and +1**.
- **Historical Sentiment Simulation:**  
  Since historical sentiment data is often unavailable, a **simulated sentiment trend** is generated (sine wave + noise around recent sentiment).
- **Merging:**  
  Combines the simulated sentiment series with the historical stock data.

---

### **3. Data Preprocessing (`_preprocess_data` function)**
- **Return Calculation:**  
  Computes daily percentage returns (`y`).
- **Outlier Handling:**  
  Applies **winsorization** (caps extreme 1% tail values).
- **Stationarity Testing:**  
  Conducts an **Augmented Dickey-Fuller (ADF)** test to verify stationarity (a requirement for ARDL).

---

### **4. Feature Engineering (Lag Creation)**
- **Distributed Lags (DL):**  
  Creates lagged features for sentiment data (`sentiment_lag_1`, ..., `sentiment_lag_q`).
- **Autoregressive Lags (AR):**  
  Creates lagged features for returns (`y_lag_1`, ..., `y_lag_p`).
- **Alignment:**  
  Combines and cleans all lag features, removing NaN values.

---

### **5. Two-Stage ARDL Modeling**
#### **Stage 1 – Mean Model**
- Fits an **ARDL(p,q)** using `statsmodels.tsa.arima.ARIMA` with exogenous sentiment lags.
- Models expected **return direction & magnitude**.
- Extracts residuals for volatility modeling.

#### **Stage 2 – Volatility Model**
- Models volatility via the **log of squared residuals** (log_vol).
- Fits another ARDL(p,q) model to capture **volatility shocks and sentiment effects**.

---

### **6. Forecasting (Configurable 10-Day Horizon)**
- **Future Sentiment:**  
  Assumes the latest sentiment persists through the forecast period.
- **Volatility Forecast:**  
  Uses Stage 2 model to predict log_vol → variance → daily volatility.
- **Mean Forecast:**  
  Uses Stage 1 model to predict returns and project future prices.
- **Transformation:**  
  Converts volatility to percentage and annualized forms.

---

### **7. AI Summarization & Recommendation (UI)**
- **Summarization:**  
  Uses **DistilBART** to summarize financial news.
- **Recommendation:**  
  Generates **Buy / Sell / Hold** signals based on recent vs. historical sentiment trends.

---

### **8. Output Generation**
- Formats forecast results including:
  - Date  
  - Predicted Price  
  - Confidence Bands  
  - Price Change  
  - Predicted Volatility  
- Saves results to `forecast_output.csv` and displays them in the UI/CLI.
- Caches Hugging Face models locally in the `./models` directory for faster reuse.

---

## 🧩 Project Structure

```

Sentiment-Driven-Volatility-Forecaster/
├── core/
│   ├── data_fetcher.py           # Stock data (yfinance)
│   ├── sentiment_analyzer.py     # NewsAPI + FinBERT + DistilBART integration
│   └── volatility_modeler.py     # 2-Stage ARDL modeling & forecasting
├── utils/
│   ├── config.py                 # API key management
│   └── logging_config.py         # Logging setup
├── models/                       # Cached Hugging Face models
├── .env                          # User-provided API key file
├── main.py                       # Command-Line Interface (CLI)
├── ui.py                         # Streamlit-based Web UI
├── requirements.txt              # Dependencies
└── README.md                     # Documentation (this file)

````

---

## ⚡ Setup Instructions

### **Prerequisites**
- Python 3.9 (recommended)
- Conda (recommended for environment management)
- NewsAPI key from [newsapi.org](https://newsapi.org)

---

### **1. Clone the Repository**
```bash
git clone https://github.com/ss1910singh/Sentiment-Driven-Volatility-Forecaster-SDVF.git
cd Sentiment-Driven-Volatility-Forecaster-SDVF
````

### **2. Create `.env` File**

```bash
echo "NEWS_API_KEY=YOUR_ACTUAL_NEWSAPI_KEY" > .env
```

### **3. Set Up Conda Environment**

```bash
conda create --name sdvf_env python=3.9
conda activate sdvf_env
```

### **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ The first run may take time to download **FinBERT** and **DistilBART** models from Hugging Face.

---

## 🚀 How to Run

### **Option 1 – Run the Web UI (Streamlit Terminal)**

```bash
streamlit run ui.py
```

Launches the **interactive AI Trading Terminal** in your browser.
Use the sidebar to:

* Enter a stock ticker (e.g., AAPL, GOOGL)
* Adjust ARDL model parameters
* Run full analysis and view AI summaries

---

### **Option 2 – Run the CLI**

```bash
python main.py <TICKER> [OPTIONS]
```

**Examples:**

```bash
python main.py AAPL
python main.py GOOGL --years 5 --ar 3 --sentiment 3 --horizon 5
```

**Available CLI Options**

| Option        | Description              | Default    |
| ------------- | ------------------------ | ---------- |
| `<TICKER>`    | Stock ticker symbol      | *Required* |
| `--years`     | Years of historical data | 10         |
| `--ar`        | Autoregressive lags (p)  | 5          |
| `--sentiment` | Sentiment lags (q)       | 5          |
| `--horizon`   | Forecast horizon (days)  | 10         |

---

## 📊 Output

**CLI:**

* Displays ARDL summaries and forecast tables.

**Web UI:**

* Interactive charts, volatility forecasts, sentiment summaries, and AI recommendations.

**File Output:**

* `forecast_output.csv` – detailed forecast data for further analysis in Excel, Power BI, etc.

---

## ⚠️ Important Notes

* **NewsAPI Limits:** Free tier limits requests per day.
* **Sentiment Simulation:** Historical sentiment is simulated for demonstration.
* **Model Downloads:** FinBERT and DistilBART are downloaded once and cached locally.
* **Performance:** Long histories or many lags may increase computation time.

---

## 🔮 Future Enhancements

* Integration of **real historical sentiment datasets**
* Advanced AI recommendation logic (volatility + technical indicators)
* **Backtesting** for evaluating forecast accuracy
* Inclusion of **evaluation metrics** (RMSE, MAE, etc.)
* Support for **custom AI model selection** via UI/CLI
* **Parallelized** model training for faster computation
* Improved **error handling** and **user feedback**

---

## 🧾 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it with attribution.