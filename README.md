# 📈 Sentiment-Driven Volatility Forecaster (SDVF)

**Integrating financial news sentiment into stock market volatility forecasts using ARX-GARCH modeling.**

---

## 🧠 Overview

The **Sentiment-Driven Volatility Forecaster (SDVF)** is an **industry-level analytics tool** that enhances traditional volatility forecasting by integrating **financial news sentiment**.  
It offers both:

- 🧩 **Command-Line Interface (CLI)** — for automation and scripting  
- 💻 **Interactive Web Interface (UI)** — built with **Streamlit** for intuitive exploration  

Standard econometric models like **GARCH** often rely solely on historical price data, ignoring market sentiment and external shocks. SDVF bridges that gap by quantifying **news sentiment** and incorporating it as an **external regressor** into an advanced **ARX-GARCH(1,1)** model.

By combining an **Autoregressive Distributed Lag (ARDL)** model for mean returns with a **GARCH(1,1)** model for volatility, SDVF produces **context-aware forecasts** that better reflect market dynamics influenced by real-world events.

---

## 🚀 Key Features

- **🧭 Dual Interface**  
  Run analysis through the command line or via an interactive **Streamlit web dashboard**.

- **📊 Advanced Financial Modeling**  
  Implements an **ARX-GARCH(1,1)** model for joint forecasting of mean returns and volatility.

- **📰 News Sentiment Analysis**  
  Uses **NLTK’s VADER** sentiment analyzer to evaluate real-time financial news fetched via **NewsAPI**.

- **🧩 Robust Architecture**  
  Modular design separating data fetching, sentiment processing, modeling, and configuration.

- **📜 Structured Logging**  
  Provides detailed logs for traceability and easier debugging.

- **🧪 Testable Components**  
  Includes unit tests to validate model and sentiment analysis components.

---

## 🧰 Prerequisites

Before installing, ensure you have:

- **Python 3.6+**
- **Git**
- **NewsAPI Key** → Get a free one from [newsapi.org](https://newsapi.org/)
- *(Optional)* **Conda** (Anaconda or Miniconda)

---

## ⚙️ Setup & Installation

You can use either **venv (Python)** or **conda** to set up the environment.

---

### 🔹 Option 1: Using venv (Standard Python)

```bash
# Clone the repository
git clone https://github.com/ss1910singh/Sentiment-Driven-Volatility-Forecaster-SDVF-.git
cd .\Sentiment-Driven-Volatility-Forecaster-SDVF-\

# Create and activate a virtual environment
# macOS/Linux
python3 -m venv venv && source venv/bin/activate
# Windows
python -m venv venv && venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the NLTK VADER lexicon
python -m nltk.downloader vader_lexicon
````

---

### 🔹 Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-driven-volatility-forecaster.git
cd sentiment-driven-volatility-forecaster

# Create and activate a conda environment
conda create --name sdvf-env python=3.9 -y
conda activate sdvf-env

# Install dependencies
pip install -r requirements.txt

# Download the NLTK VADER lexicon
python -m nltk.downloader vader_lexicon
```

---

## ▶️ Usage

Ensure your environment (`venv` or `conda`) is activated before running.

---

### 🖥️ Running the Web UI (Recommended)

Launch the interactive dashboard with:

```bash
streamlit run ui.py
```

Your browser will open automatically with the **SDVF dashboard**.
Enter:

* **Stock tickers** (e.g., `AAPL`, `GOOG`)
* **Your NewsAPI key**

Then click **“Run Forecast”** to visualize:

* Sentiment over time
* Volatility predictions
* Model diagnostics

---

### 💻 Running via Command-Line Interface (CLI)

Ideal for automation, scripting, or headless environments.

```bash
python main.py --tickers AAPL GOOG --api-key YOUR_NEWS_API_KEY_HERE
```

Example Output:

```
[INFO] Fetching stock data for AAPL, GOOG...
[INFO] Performing sentiment analysis...
[INFO] Training ARX-GARCH(1,1) model...
[RESULT] AAPL: Forecasted Volatility (Next 5 Days): 2.37%
[RESULT] GOOG: Forecasted Volatility (Next 5 Days): 3.05%
```

To deactivate the environment:

```bash
deactivate
# or
conda deactivate
```

---

## ⚠️ Important Note on Historical Sentiment Data

The **free tier** of NewsAPI does **not** provide access to historical news archives.
Hence, SDVF **simulates historical sentiment data** to demonstrate ARDL integration.

For a **production-grade system**, you can integrate real historical sentiment sources like:

* [Alpha Vantage](https://www.alphavantage.co/)
* [Refinitiv](https://www.refinitiv.com/)
* Custom **archival scrapers or APIs**

---

## 🧩 Project Structure

```
sentiment-driven-volatility-forecaster/
│
├── core/
│   ├── data_fetcher.py          # Fetches historical stock price data
│   ├── sentiment_analyzer.py    # Analyzes financial news sentiment
│   └── volatility_modeler.py    # Implements ARX-GARCH model
│
├── utils/
│   ├── config.py                # Handles configuration (API keys, settings)
│   └── logging_config.py        # Sets up structured logging
│
├── tests/
│   └── test_sentiment_analyzer.py   # Unit tests for sentiment analysis
│
├── main.py                      # CLI entry point
├── ui.py                        # Web UI (Streamlit) entry point
├── requirements.txt             # Dependencies list
└── README.md                    # Documentation
```

---

## 🧪 Testing

Run the unit tests using:

```bash
pytest tests/
```

---

## 🧾 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

> “Markets move not only by numbers — but by narratives. SDVF quantifies both.”