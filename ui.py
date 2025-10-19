import streamlit as st
import pandas as pd
from core.data_fetcher import fetch_stock_data
from core.sentiment_analyzer import fetch_recent_news_sentiment, simulate_historical_sentiment
from core.volatility_modeler import fit_garch_model_and_forecast

st.set_page_config(
    page_title="Sentiment-Driven Volatility Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sentiment-Driven Volatility Forecaster (SDVF)")
st.markdown("""
    This tool enhances traditional stock market volatility forecasts by incorporating financial news sentiment. 
    Enter a stock ticker to analyze its historical performance and get a 5-day volatility forecast.
""")

st.sidebar.header("User Input")
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "AAPL")
analyze_button = st.sidebar.button("Analyze Volatility")

if analyze_button:
    if not ticker_input:
        st.error("Please enter a stock ticker.")
    else:
        ticker = ticker_input.upper()
        
        with st.spinner(f"Running analysis for {ticker}... This may take a moment."):
            try:
                stock_data = fetch_stock_data(ticker)
                if stock_data.empty:
                    raise ValueError(f"Could not fetch data for ticker '{ticker}'. Please check if the ticker is valid.")
                
                recent_sentiment = fetch_recent_news_sentiment(ticker)
                stock_data_with_sentiment = simulate_historical_sentiment(stock_data.copy())
                forecast_df = fit_garch_model_and_forecast(stock_data_with_sentiment)
                st.header(f"Analysis for {ticker}")

                col1, col2, col3 = st.columns(3)
                last_close_price = stock_data['Close'].iloc[-1]
                col1.metric("Last Close Price", f"${last_close_price:,.2f}")
                col2.metric("Recent News Sentiment", f"{recent_sentiment:.4f}", help="Score from -1 (Negative) to +1 (Positive)")
                
                if not forecast_df.empty:
                    avg_forecast_vol = forecast_df['Predicted Annualized Volatility (%)'].mean()
                    col3.metric("Avg. Forecast Volatility", f"{avg_forecast_vol:.2f}%", help="Average predicted annualized volatility for the next 5 days.")

                st.subheader("Historical Data & Sentiment")
                st.line_chart(stock_data_with_sentiment[['Close', 'sentiment']], use_container_width=True)
                st.caption("Historical closing prices and simulated historical sentiment score.")

                if not forecast_df.empty:
                    st.subheader("Volatility Forecast (Next 5 Trading Days)")
                    st.dataframe(forecast_df)
                    st.info("The forecast data has also been saved to `forecast_output.csv`.")
                    
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Forecast Data for Power BI",
                        data=csv,
                        file_name=f"{ticker}_forecast.csv",
                        mime='text/csv',
                    )
                else:
                    st.error("Failed to generate a forecast. The GARCH model could not converge. This can happen with highly stable or unpredictable stocks. Please check logs for details.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

else:
    st.info("Enter a stock ticker in the sidebar and click 'Analyze Volatility' to begin.")