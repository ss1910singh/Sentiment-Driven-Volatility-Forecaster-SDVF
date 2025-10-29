import streamlit as st
import pandas as pd
import altair as alt
import logging
import numpy as np

from core.data_fetcher import get_stock_data
from core.sentiment_analyzer import add_sentiment_to_data, summarize_articles, get_buy_sell_recommendation
from core.volatility_modeler import fit_ardl_volatility_model
from utils.logging_config import setup_logging

st.set_page_config(
    page_title="AI Trading Terminal (ARDL Model)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging()

def format_change(change):
    if pd.isna(change): return "—"
    if change > 0:
        return f"▲ {change:.2f}"
    elif change < 0:
        return f"▼ {change:.2f}"
    else:
        return f"— {change:.2f}"

st.title("🤖 AI-Powered Trading Terminal")
st.markdown("""
Forecasts volatility and price using a **2-Stage ARDL (Autoregressive Distributed Lag)** model,
powered by Hugging Face transformers for financial sentiment analysis and news summarization.
""")

with st.sidebar:
    st.header("🛠️ Model Configuration")

    ticker = st.text_input(
        "Enter Stock Ticker (e.g., AAPL, GOOGL)",
        "AAPL"
    ).upper()

    st.markdown("### ARDL Model Parameters")
    years_of_data = st.slider(
        "Years of Historical Data",
        min_value=1,
        max_value=10,
        value=10,
        help="Number of years of data to use for training the model."
    )
    ar_lags = st.slider(
        "Autoregressive (AR) Lags (p)",
        min_value=0,
        max_value=10,
        value=5,
        help="Number of past *return* lags to use in the model."
    )
    sentiment_lags = st.slider(
        "Sentiment (DL) Lags (q)",
        min_value=0,
        max_value=10,
        value=5,
        help="Number of past *sentiment* lags to use in the model."
    )
    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=30,
        value=10,
        help="Number of days to forecast into the future."
    )

    run_button = st.button("Run Full Analysis", type="primary", use_container_width=True)


if run_button:
    st.empty()

    data_placeholder = st.empty()
    with data_placeholder.status(f"Fetching {years_of_data} years of stock data for {ticker}...", expanded=True):
        stock_data = get_stock_data(ticker, years=years_of_data)

    if stock_data.empty:
        data_placeholder.error(f"Could not fetch stock data for {ticker}. Please check the ticker and try again.")
        st.stop()
    else:
        data_placeholder.success(f"Successfully fetched {len(stock_data)} data points for {ticker}.")

    sentiment_placeholder = st.empty()
    with sentiment_placeholder.status(f"Fetching and analyzing recent news for {ticker} (using FinBERT)...", expanded=True):
        try:
            stock_data_with_sentiment, recent_sentiment, recent_articles = add_sentiment_to_data(stock_data, ticker)
            recommendation, reasoning = get_buy_sell_recommendation(stock_data_with_sentiment['sentiment'], recent_sentiment)
            st.success("Sentiment analysis complete.")
        except Exception as e:
            sentiment_placeholder.error(f"Failed to get news sentiment: {e}. Check your NEWS_API_KEY and AI model setup.")
            st.stop()

    delta_text = ""
    if recommendation == "Buy" or recommendation == "Strong Buy":
        delta_text = "▲ Positive Sentiment Trend"
    elif recommendation == "Sell":
        delta_text = "▼ Negative Sentiment Trend"
    else:
         delta_text = "Neutral Sentiment Trend"

    st.metric(label=f"AI Recommendation for {ticker}", value=recommendation, delta=delta_text)
    st.caption(reasoning)
    st.divider()

    tab1, tab2 = st.tabs(["📈 Forecast & Charts", "📰 News & AI Summaries"])

    with st.spinner(f"Fitting 2-Stage ARDL models and generating {forecast_horizon}-day forecast..."):
        try:
            final_forecast_df, fitted_mean_values = fit_ardl_volatility_model(
                stock_data_with_sentiment,
                recent_sentiment,
                ar_lags=ar_lags,
                sentiment_lags=sentiment_lags,
                forecast_horizon=forecast_horizon
            )
        except Exception as e:
            st.error(f"An error occurred during modeling: {e}")
            logging.error(f"Modeling error: {e}", exc_info=True)
            final_forecast_df = pd.DataFrame()
            fitted_mean_values = pd.Series(dtype=float)

    with tab1:
        if final_forecast_df is None or fitted_mean_values is None or final_forecast_df.empty or fitted_mean_values.empty:
             st.error("Model fitting failed or returned empty results. Could not generate chart or table. Please check logs.")
        else:
            st.subheader("📈 Advanced Forecast Chart (Price)")
            st.markdown("This chart shows the actual historical price, the model's *fitted* values (convergence), and the *forecasted* price (prediction) with confidence bands.")

            last_actual_date = pd.to_datetime(stock_data_with_sentiment.index[-1]).tz_localize(None)
            last_actual_price = stock_data_with_sentiment['Close'].iloc[-1]

            display_years = min(years_of_data, 3)
            start_date_chart = last_actual_date - pd.DateOffset(years=display_years)
            hist_index_naive = pd.to_datetime(stock_data_with_sentiment.index).tz_localize(None)
            hist_data_full = stock_data_with_sentiment[hist_index_naive >= start_date_chart].reset_index()
            if 'index' in hist_data_full.columns: hist_data_full = hist_data_full.rename(columns={'index':'Date'})
            if 'Date' not in hist_data_full.columns:
                 date_col_name = hist_data_full.columns[0]
                 hist_data_full = hist_data_full.rename(columns={date_col_name: 'Date'})

            hist_data_full['Date'] = pd.to_datetime(hist_data_full['Date']).dt.tz_localize(None)
            hist_data = hist_data_full[['Date', 'Close']].copy()
            hist_data['Type'] = 'Actual'


            fitted_data = pd.DataFrame()
            try:
                stock_idx_naive = pd.to_datetime(stock_data_with_sentiment.index).tz_localize(None)
                fitted_idx_naive = pd.to_datetime(fitted_mean_values.index).tz_localize(None)
                stock_data_naive = stock_data_with_sentiment.copy()
                stock_data_naive.index = stock_idx_naive
                aligned_fitted_returns = fitted_mean_values.copy()
                aligned_fitted_returns.index = fitted_idx_naive
                aligned_fitted_returns = aligned_fitted_returns.reindex(stock_idx_naive).ffill()
                fitted_prices = stock_data_naive['Close'].shift(1) * (1 + aligned_fitted_returns / 100)
                fitted_prices_chart = fitted_prices[fitted_prices.index >= start_date_chart].dropna()

                if not fitted_prices_chart.empty:
                    fitted_data = pd.DataFrame({
                        'Date': fitted_prices_chart.index,
                        'Price': fitted_prices_chart,
                        'Type': 'Fitted (In-Sample)'
                    })
                else:
                    logging.warning("No fitted data points fall within the chart's display window after calculation.")
                    st.warning("Warning: No fitted data points fall within the chart's display window.")

            except Exception as e_fit:
                 logging.error(f"Error calculating fitted data for chart: {e_fit}", exc_info=True)
                 st.warning("Could not calculate or display fitted data points due to an alignment error.")


            forecast_data = final_forecast_df[['Date', 'Predicted Price', 'Lower Band', 'Upper Band']].copy()
            forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.tz_localize(None)
            forecast_data = forecast_data.rename(columns={'Predicted Price': 'Price'})
            forecast_data['Type'] = 'Forecast (Out-of-Sample)'

            last_actual_df = pd.DataFrame({
                'Date': [last_actual_date],
                'Price': [last_actual_price],
                'Lower Band': [last_actual_price],
                'Upper Band': [last_actual_price],
                'Type': ['Forecast (Out-of-Sample)']
            })

            last_actual_df['Date'] = pd.to_datetime(last_actual_df['Date'])
            forecast_data = pd.concat([last_actual_df, forecast_data]).sort_values('Date')
            plot_data = pd.concat([hist_data, fitted_data, forecast_data]).dropna(subset=['Price'])
            plot_data['Date'] = pd.to_datetime(plot_data['Date'])


            if plot_data.empty:
                st.warning("No data available to plot after processing. Check model results and data alignment.")
            else:
                base = alt.Chart(plot_data).encode(x=alt.X('Date', axis=alt.Axis(title='Date')))
                price_lines = base.mark_line(point=False).encode(
                    y=alt.Y('Price', title='Stock Price ($)', axis=alt.Axis(format='$,.2f')),
                    color=alt.Color('Type', title='Data Type',
                                    scale=alt.Scale(domain=['Actual', 'Fitted (In-Sample)', 'Forecast (Out-of-Sample)'],
                                                    range=['#1f77b4', '#ff7f0e', '#2ca02c']),
                                    legend=alt.Legend(orient="bottom")),
                    tooltip=['Date', alt.Tooltip('Price', format='$,.2f'), 'Type']
                )

                confidence_band = base.mark_area(opacity=0.3, color='#2ca02c').encode(
                    y=alt.Y('Lower Band', axis=alt.Axis(format='$,.2f')),
                    y2=alt.Y2('Upper Band'),
                    tooltip=[alt.Tooltip('Lower Band', format='$,.2f', title='Lower 95% CI'),
                             alt.Tooltip('Upper Band', format='$,.2f', title='Upper 95% CI')]
                ).transform_filter(
                    alt.datum.Type == 'Forecast (Out-of-Sample)'
                )

                final_chart = (price_lines + confidence_band).interactive()
                st.altair_chart(final_chart, use_container_width=True)
                st.caption("Forecast includes a 95% confidence interval based on predicted volatility.")

            st.divider()
            st.subheader(f"📊 {forecast_horizon}-Day Advanced Forecast Table")

            display_df = final_forecast_df.copy()
            display_df['Price Change'] = display_df['Price Change'].apply(format_change)

            column_config = {
                 "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                 "Predicted Price": st.column_config.NumberColumn("Predicted Price ($)", format="$%.2f"),
                 "Price Change": st.column_config.TextColumn("Price Change ($)"),
                 "Lower Band": st.column_config.NumberColumn("Lower 95% CI ($)", format="$%.2f"),
                 "Upper Band": st.column_config.NumberColumn("Upper 95% CI ($)", format="$%.2f"),
                 "Predicted Daily Volatility (%)": st.column_config.NumberColumn("Daily Vol (%)", format="%.3f%%"),
                 "Predicted Annualized Volatility (%)": st.column_config.NumberColumn("Annual Vol (%)", format="%.3f%%"),
                 "Predicted Return (%)": st.column_config.NumberColumn("Return (%)", format="%.3f%%"), # Added Return %
            }

            output_columns = [
                "Date",
                "Predicted Price",
                "Price Change",
                "Predicted Return (%)",
                "Lower Band",
                "Upper Band",
                "Predicted Daily Volatility (%)",
                "Predicted Annualized Volatility (%)"
            ]

            st.dataframe(
                display_df[output_columns],
                column_config=column_config,
                hide_index=True,
                use_container_width=True
             )

    with tab2:
        with st.spinner("Summarizing articles with AI..."):
            try:
                if 'recent_articles' in locals() and recent_articles:
                    individual_summaries, combined_summary = summarize_articles(recent_articles)

                    st.subheader("📰 Combined AI Summary of Recent News")
                    st.markdown(combined_summary)
                    st.divider()

                    st.subheader(f"📑 Individual Article Summaries ({len(individual_summaries)})")
                    if not individual_summaries:
                        st.write("No articles could be summarized (might be too short or unavailable).")
                    else:
                        cols = st.columns(2)
                        col_index = 0
                        for i, item in enumerate(individual_summaries):
                            with cols[col_index % 2]:
                               with st.container(border=True):
                                    st.markdown(f"**{item['title']}**")
                                    st.caption(f"Source: {item['source']}")
                                    st.markdown(item['summary'])
                                    st.link_button("Read Full Article ↗️", item['url'], use_container_width=True)
                            col_index += 1
                else:
                    st.warning("No recent articles were found to summarize.")

            except Exception as e:
                st.error(f"Failed to summarize articles: {e}")
                logging.error(f"Summarization error: {e}", exc_info=True)