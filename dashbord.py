import streamlit as st
import pandas as pd
import altair as alt
import logging
import numpy as np
import os
import glob
import plotly.graph_objects as go

from core.data_fetcher import get_stock_data
from core.sentiment_analyzer import add_sentiment_to_data, summarize_articles, get_buy_sell_recommendation
from core.volatility_modeler import fit_ardl_volatility_model
from utils.logging_config import setup_logging

st.set_page_config(
    page_title="AI Trading Insights Dashboard",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

setup_logging()
CSV_DATA_DIR = "./Fetch_data/data"
os.makedirs(CSV_DATA_DIR, exist_ok=True)

def format_change(change):
    if pd.isna(change): return "—"
    if change > 0: return f"▲ {change:.2f}"
    elif change < 0: return f"▼ {change:.2f}"
    else: return f"— {change:.2f}"

@st.cache_data
def load_csv_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [col.strip().lower() for col in df.columns]
        if 'date' not in df.columns:
            st.error(f"CSV '{os.path.basename(csv_path)}' must have a 'date' column.")
            return None
        try: df['date'] = pd.to_datetime(df['date'])
        except ValueError as date_err:
             st.error(f"Error parsing 'date' in {os.path.basename(csv_path)}: {date_err}")
             return None
        df = df.sort_values('date').set_index('date')
        required_cols = ['close', 'high', 'low', 'open', 'volume']
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 df[col] = np.nan

        missing_cols = [col for col in required_cols if col in df.columns and df[col].isnull().all()] 
        if missing_cols: st.warning(f"CSV missing or has invalid data in columns: {', '.join(missing_cols)}.")

        if 'close' in df.columns and df['close'].notna().any():
             df['close'] = df['close'].astype(float)
             df['daily_return'] = df['close'].pct_change() * 100
        else:
            df['daily_return'] = np.nan

        df = df.dropna(subset=['close'])
        if df.empty:
             st.error(f"No valid 'close' data found in {os.path.basename(csv_path)} after cleaning.")
             return None
        return df
    except FileNotFoundError: st.error(f"CSV file not found: {csv_path}"); return None
    except Exception as e: st.error(f"Error loading CSV '{os.path.basename(csv_path)}': {e}"); return None

default_values = {
    'view': "Live Analysis", 'ticker': "AAPL", 'years_of_data': 10,
    'ar_lags': 5, 'sentiment_lags': 5, 'forecast_horizon': 10,
    'results_live': None, 'selected_csv': None
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Stocks Dashboard")

view_options = ["Live Analysis", "Historical Analysis"]
def view_change_callback():
     if st.session_state.view_selector_radio != "Live Analysis":
         st.session_state.results_live = None
     st.session_state.view = st.session_state.view_selector_radio

selected_view_from_radio = st.radio(
    "Select View:", view_options,
    index=view_options.index(st.session_state.view),
    horizontal=True, key='view_selector_radio',
    label_visibility='collapsed', on_change=view_change_callback
)

st.divider()


def render_live_analysis_page():
    """Renders the main AI Trading Terminal page."""
    st.header(f"Live Analysis & Forecasting: {st.session_state.ticker}")

    with st.expander("Configure Live Analysis", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
             def ticker_change_callback(): st.session_state.results_live = None
             st.session_state.ticker = st.text_input("Ticker", st.session_state.ticker, key="live_ticker_input", on_change=ticker_change_callback).upper()
        with col2:
             st.session_state.years_of_data = st.number_input(
                 "Years Data", min_value=1, max_value=20,
                 value=st.session_state.years_of_data, step=1, key="live_years_input"
             )
        with col3:
             st.session_state.ar_lags = st.number_input(
                 "AR Lags (p)", min_value=0, max_value=20,
                 value=st.session_state.ar_lags, step=1, key="live_ar_input"
            )
        with col4:
             st.session_state.sentiment_lags = st.number_input(
                 "Sent. Lags (q)", min_value=0, max_value=20,
                 value=st.session_state.sentiment_lags, step=1, key="live_sent_input"
             )
        with col5:
             st.session_state.forecast_horizon = st.number_input(
                 "Forecast Days", min_value=1, max_value=90,
                 value=st.session_state.forecast_horizon, step=1, key="live_horizon_input"
             )

    ticker = st.session_state.ticker
    years_of_data = st.session_state.years_of_data
    ar_lags = st.session_state.ar_lags
    sentiment_lags = st.session_state.sentiment_lags
    forecast_horizon = st.session_state.forecast_horizon

    if st.button(f"▶️ Run Live Analysis for {ticker}", type="primary", use_container_width=True, key="live_run_button"):
        st.session_state.results_live = None
        with st.spinner(f"Running full analysis for {ticker}... (This may take a minute)"):
            try:
                logging.info(f"Starting analysis for {ticker}...")
                logging.info(f"Fetching {years_of_data}y data...")
                stock_data = get_stock_data(ticker, years=years_of_data)
                if stock_data is None or stock_data.empty: raise ValueError("Error fetching data.")
                logging.info(f"Data fetched ({len(stock_data)} points). Analyzing news...")

                stock_data_with_sentiment, recent_sentiment, recent_articles = add_sentiment_to_data(stock_data.copy(), ticker)
                if stock_data_with_sentiment is None: raise ValueError("Sentiment analysis failed.")
                recommendation, reasoning = get_buy_sell_recommendation(stock_data_with_sentiment['sentiment'], recent_sentiment)
                logging.info("Sentiment analyzed. Fitting models...")

                final_forecast_df, fitted_mean_values = fit_ardl_volatility_model(
                    stock_data_with_sentiment, recent_sentiment, ar_lags, sentiment_lags, forecast_horizon
                )
                if final_forecast_df is None or fitted_mean_values is None or final_forecast_df.empty or fitted_mean_values.empty:
                    raise ValueError("Model fitting returned empty results.")
                logging.info("Models fit & forecast generated.")

                st.session_state.results_live = {
                    "ticker": ticker, "stock_data_with_sentiment": stock_data_with_sentiment,
                    "recent_sentiment": recent_sentiment, "recent_articles": recent_articles,
                    "recommendation": recommendation, "reasoning": reasoning,
                    "final_forecast_df": final_forecast_df, "fitted_mean_values": fitted_mean_values,
                    "display_years": min(years_of_data, 3)
                }
                st.success("Live analysis complete!")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logging.error(f"Live analysis failed: {e}", exc_info=True)
                st.session_state.results_live = None

    if st.session_state.get("results_live"):
        results = st.session_state.results_live
        if results['ticker'] != ticker:
             st.info(f"Showing previous results for {results['ticker']}. Click 'Run Live Analysis' for {ticker}.")
        else:
            current_ticker_results = results['ticker']
            st.markdown(f"### Key Metrics for {current_ticker_results}")
            col1, col2, col3 = st.columns(3)
            last_close = results['stock_data_with_sentiment']['Close'].iloc[-1] if not results['stock_data_with_sentiment'].empty else np.nan
            col1.metric("Last Close", f"${last_close:,.2f}" if pd.notna(last_close) else "N/A")
            col2.metric("Recent Sentiment", f"{results['recent_sentiment']:.4f}")
            delta_map = {"Buy": "▲ Positive", "Strong Buy": "▲ Positive", "Sell": "▼ Negative"}
            col3.metric("AI Recommendation", results['recommendation'], delta=delta_map.get(results['recommendation'], "Neutral") + " Trend")
            st.caption(results['reasoning'])
            st.divider()

            left_col, right_col = st.columns([2, 1])
            with left_col:
                st.subheader("📈 Forecast & Charts")
                try:
                    stock_data_ws = results.get('stock_data_with_sentiment')
                    fitted_mv = results.get('fitted_mean_values')
                    final_fc_df = results.get('final_forecast_df')
                    display_yrs = results.get('display_years', 3)

                    if not all([isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty for df in [stock_data_ws, fitted_mv, final_fc_df]]):
                       st.warning("Chart data incomplete.")
                    else:
                        last_actual_date = pd.to_datetime(stock_data_ws.index[-1]).tz_localize(None)
                        last_actual_price = stock_data_ws['Close'].iloc[-1]
                        start_date_chart = last_actual_date - pd.DateOffset(years=display_yrs)
                        hist_idx_naive = pd.to_datetime(stock_data_ws.index).tz_localize(None)
                        hist_data_full = stock_data_ws[hist_idx_naive >= start_date_chart].reset_index()
                        date_col = hist_data_full.columns[0]; hist_data_full = hist_data_full.rename(columns={date_col: 'Date'})
                        hist_data_full['Date'] = pd.to_datetime(hist_data_full['Date']).dt.tz_localize(None)
                        hist_data = hist_data_full[['Date', 'Close']].copy(); hist_data['Type'] = 'Actual'
                        fitted_data = pd.DataFrame()
                        stock_idx_naive = pd.to_datetime(stock_data_ws.index).tz_localize(None)
                        fitted_idx_naive = pd.to_datetime(fitted_mv.index).tz_localize(None)
                        stock_data_naive = stock_data_ws.copy(); stock_data_naive.index = stock_idx_naive
                        aligned_fit_ret = fitted_mv.copy(); aligned_fit_ret.index = fitted_idx_naive
                        aligned_fit_ret = aligned_fit_ret.reindex(stock_idx_naive).ffill()
                        close_shifted = stock_data_naive['Close'].shift(1)
                        valid_idx = close_shifted.index.intersection(aligned_fit_ret.index)
                        fitted_prices = close_shifted.loc[valid_idx] * (1 + aligned_fit_ret.loc[valid_idx] / 100)
                        fitted_prices_chart = fitted_prices[fitted_prices.index >= start_date_chart].dropna()
                        if not fitted_prices_chart.empty:
                            fitted_data = pd.DataFrame({'Date': fitted_prices_chart.index, 'Price': fitted_prices_chart, 'Type': 'Fitted (In-Sample)'})
                        forecast_data = final_fc_df[['Date', 'Predicted Price', 'Lower Band', 'Upper Band']].copy()
                        forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.tz_localize(None)
                        forecast_data = forecast_data.rename(columns={'Predicted Price': 'Price'})
                        forecast_data['Type'] = 'Forecast (Out-of-Sample)'
                        last_actual_df = pd.DataFrame({'Date': [last_actual_date], 'Price': [last_actual_price], 'Lower Band': [last_actual_price], 'Upper Band': [last_actual_price], 'Type': ['Forecast (Out-of-Sample)']})
                        last_actual_df['Date'] = pd.to_datetime(last_actual_df['Date'])
                        forecast_data = pd.concat([last_actual_df, forecast_data]).sort_values('Date')
                        plot_data = pd.concat([hist_data, fitted_data, forecast_data]).dropna(subset=['Price'])
                        plot_data['Date'] = pd.to_datetime(plot_data['Date'])

                        if plot_data.empty: st.warning("No data available to plot.")
                        else:
                            base = alt.Chart(plot_data).encode(x=alt.X('Date', axis=alt.Axis(title='Date')))
                            price_lines = base.mark_line(point=False).encode(
                                y=alt.Y('Price', title='Stock Price ($)', axis=alt.Axis(format='$,.2f')),
                                color=alt.Color('Type', title='Data Type',
                                                scale=alt.Scale(domain=['Actual', 'Fitted (In-Sample)', 'Forecast (Out-of-Sample)'],
                                                                range=['#1f77b4', '#ff7f0e', '#2ca02c']), # Blue, Orange, Green
                                                legend=alt.Legend(orient="bottom")),
                                tooltip=[alt.Tooltip('Date', format='%Y-%m-%d'), alt.Tooltip('Price', format='$,.2f'), 'Type']
                            )
                            confidence_band = base.mark_area(opacity=0.3, color='#2ca02c').encode(
                                y=alt.Y('Lower Band', axis=alt.Axis(format='$,.2f')), y2=alt.Y2('Upper Band'),
                                tooltip=[alt.Tooltip('Lower Band', format='$,.2f', title='Lower 95% CI'), alt.Tooltip('Upper Band', format='$,.2f', title='Upper 95% CI')]
                            ).transform_filter(alt.datum.Type == 'Forecast (Out-of-Sample)')
                            final_chart = (price_lines + confidence_band).interactive()
                            st.altair_chart(final_chart, use_container_width=True)
                            st.caption("Forecast includes a 95% confidence interval.")

                except Exception as chart_e: st.error(f"Error generating forecast chart: {chart_e}"); logging.error(f"Charting error: {chart_e}", exc_info=True)

                st.divider()
                st.subheader(f"📊 {len(final_forecast_df)}-Day Forecast Table")
                display_df = final_forecast_df.copy()
                display_df['Price Change'] = display_df['Price Change'].apply(format_change)
                column_config = {
                     "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"), "Predicted Price": st.column_config.NumberColumn("Predicted ($)", format="$%.2f"),
                     "Price Change": st.column_config.TextColumn("Change ($)"), "Lower Band": st.column_config.NumberColumn("Lower CI ($)", format="$%.2f"),
                     "Upper Band": st.column_config.NumberColumn("Upper CI ($)", format="$%.2f"), "Predicted Daily Volatility (%)": st.column_config.NumberColumn("Daily Vol (%)", format="%.3f%%"),
                     "Predicted Annualized Volatility (%)": st.column_config.NumberColumn("Annual Vol (%)", format="%.3f%%"), "Predicted Return (%)": st.column_config.NumberColumn("Return (%)", format="%.3f%%"),
                }
                output_columns = list(column_config.keys())
                st.dataframe(display_df[output_columns], column_config=column_config, hide_index=True, use_container_width=True)

            with right_col:
                st.subheader("📰 AI News Analysis")
                with st.spinner("Summarizing articles..."):
                    try:
                        if 'recent_articles' in results and results['recent_articles']:
                            individual_summaries, combined_summary = summarize_articles(results['recent_articles'])
                            st.markdown("**Combined Summary:**")
                            st.info(combined_summary)
                            st.divider()
                            st.markdown(f"**Individual Summaries ({len(individual_summaries)}):**")
                            if not individual_summaries: st.write("No articles summarized.")
                            else:
                                with st.container(height=600):
                                     for item in individual_summaries:
                                         with st.container(border=True):
                                             st.markdown(f"**{item['title']}**"); st.caption(f"Source: {item['source']}")
                                             st.markdown(item['summary'])
                                             st.link_button("Read ↗️", item['url'])
                        else: st.warning("No recent articles found.")
                    except Exception as e: st.error(f"Failed to summarize: {e}"); logging.error(f"Summarization error: {e}", exc_info=True)

    if not st.session_state.get("results_live"):
         st.info("Configure analysis parameters in the expander above and click 'Run Live Analysis'.")

def render_historical_analysis_page():
    """Renders the historical analysis dashboard."""
    st.header("📊 Historical Data Analysis")

    try:
        csv_files = glob.glob(os.path.join(CSV_DATA_DIR, "*.csv"))
        csv_file_names = sorted([os.path.basename(f) for f in csv_files])
    except Exception as e:
        st.error(f"Error accessing CSV directory '{CSV_DATA_DIR}': {e}"); csv_file_names = []

    if not csv_file_names:
        st.warning(f"No CSV files found in '{CSV_DATA_DIR}'. Add CSV files for analysis.")
        return

    valid_options = csv_file_names
    current_selection = st.session_state.get("selected_csv", valid_options[0] if valid_options else None)
    if current_selection not in valid_options and valid_options: current_selection = valid_options[0]
    elif not valid_options: current_selection = None

    selected_csv_name = st.selectbox(
        "Select Historical Dataset:", valid_options,
        index=valid_options.index(current_selection) if current_selection in valid_options else 0,
        key="csv_selector_hist"
    )

    if selected_csv_name != st.session_state.get("selected_csv"):
         st.session_state.selected_csv = selected_csv_name
         st.rerun()

    if st.session_state.selected_csv:
        selected_csv_path = os.path.join(CSV_DATA_DIR, st.session_state.selected_csv)
        st.markdown(f"### Analyzing: {st.session_state.selected_csv.replace('.csv','')}")
        st.divider()

        df = load_csv_data(selected_csv_path)
        if df is None or df.empty: return

        st.markdown("##### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        latest_data = df.iloc[-1]
        close_val = latest_data.get('close', np.nan)
        close_display = f"${close_val:,.2f}" if pd.notna(close_val) and isinstance(close_val, (int, float)) else "N/A"
        col1.metric("Latest Close", close_display)
        prev_close = df['close'].iloc[-2] if len(df) > 1 and 'close' in df.columns and pd.notna(df['close'].iloc[-2]) else np.nan
        change = close_val - prev_close if pd.notna(close_val) and pd.notna(prev_close) else np.nan
        change_pct = (change / prev_close) * 100 if pd.notna(prev_close) and prev_close != 0 and pd.notna(change) else np.nan
        col2.metric("1-Day Change", f"{change:,.2f}" if pd.notna(change) else "N/A", delta=f"{change_pct:.2f}%" if pd.notna(change_pct) else None, delta_color=("inverse" if pd.notna(change_pct) and change_pct<0 else "normal"))
        max_high = df['high'].max() if 'high' in df.columns and pd.notna(df['high']).any() else np.nan
        col3.metric("Period Max High", f"${max_high:,.2f}" if pd.notna(max_high) else "N/A")
        avg_vol = df['volume'].mean() if 'volume' in df.columns and pd.notna(df['volume']).any() else np.nan
        col4.metric("Period Avg Volume", f"{avg_vol:,.0f}" if pd.notna(avg_vol) else "N/A")
        st.divider()

        st.markdown("##### Visualizations")
        chart_col1, chart_col2 = st.columns(2)
        df_reset = df.reset_index()

        with chart_col1:
            st.subheader("📈 Price Trend w/ Moving Averages")
            if 'close' in df.columns and pd.notna(df['close']).any():
                 df_reset['MA20'] = df_reset['close'].rolling(window=20).mean()
                 df_reset['MA50'] = df_reset['close'].rolling(window=50).mean()
                 base = alt.Chart(df_reset.dropna(subset=['close'])).encode(x=alt.X('date', axis=alt.Axis(title='Date')))
                 price_line = base.mark_line().encode(y=alt.Y('close', axis=alt.Axis(title='Price', format='$,.2f')), tooltip=[alt.Tooltip('date', format='%Y-%m-%d'), alt.Tooltip('close', format='$,.2f')], color=alt.value("#1f77b4"))
                 ma20_line = base.mark_line(opacity=0.8).encode(y=alt.Y('MA20'), tooltip=[alt.Tooltip('date', format='%Y-%m-%d'), alt.Tooltip('MA20', format='$,.2f')], color=alt.value("orange"))
                 ma50_line = base.mark_line(opacity=0.8).encode(y=alt.Y('MA50'), tooltip=[alt.Tooltip('date', format='%Y-%m-%d'), alt.Tooltip('MA50', format='$,.2f')], color=alt.value("red"))
                 price_chart = alt.layer(
                     price_line.encode(color=alt.value("#1f77b4")),
                     ma20_line.encode(color=alt.value("orange")),
                     ma50_line.encode(color=alt.value("red"))
                 ).properties(title="Closing Price & Moving Averages").interactive()
                 legend_data = pd.DataFrame({'label': ['Close', 'MA(20)', 'MA(50)'], 'color': ['#1f77b4', 'orange', 'red']})
                 legend = alt.Chart(legend_data).mark_point(size=100, filled=True).encode(y=alt.Y('label', axis=None), color=alt.Color('color', scale=None))
                 text = legend.mark_text(align='left', dx=10).encode(text='label')

                 st.altair_chart(price_chart, use_container_width=True)

            else: st.warning("Close column missing/empty.")

            st.subheader("📉 Rolling Volatility (30-Day Std Dev)")
            if 'daily_return' in df.columns and pd.notna(df['daily_return']).any():
                 df_reset['volatility'] = df_reset['daily_return'].rolling(window=30).std() * np.sqrt(252) # Annualized
                 vol_chart = alt.Chart(df_reset.dropna(subset=['volatility'])).mark_line(color='purple').encode(
                     x=alt.X('date', axis=alt.Axis(title='Date')),
                     y=alt.Y('volatility', axis=alt.Axis(title='Annualized Volatility (%)')),
                     tooltip=[alt.Tooltip('date', format='%Y-%m-%d'), alt.Tooltip('volatility', format='.2f')]
                 ).properties(title="30-Day Rolling Annualized Volatility").interactive()
                 st.altair_chart(vol_chart, use_container_width=True)
            else: st.warning("Could not calculate daily returns for volatility.")


        with chart_col2:
            st.subheader("🕯️ Candlestick Chart")
            ohlc_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in ohlc_cols) and all(pd.api.types.is_numeric_dtype(df[col]) for col in ohlc_cols):
                plot_df_ohlc = df_reset.dropna(subset=ohlc_cols).tail(180) # Last ~6 months
                if not plot_df_ohlc.empty:
                    try:
                         if not all(pd.api.types.is_numeric_dtype(plot_df_ohlc[col]) for col in ohlc_cols):
                              raise ValueError("OHLC columns non-numeric after cleaning.")
                         fig = go.Figure(data=[go.Candlestick(x=plot_df_ohlc['date'],
                                        open=plot_df_ohlc['open'], high=plot_df_ohlc['high'],
                                        low=plot_df_ohlc['low'], close=plot_df_ohlc['close'])])
                         fig.update_layout(title='Candlestick (Last ~6 Months)', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False, height=400)
                         st.plotly_chart(fig, use_container_width=True)
                    except Exception as plot_e: st.error(f"Candlestick chart error: {plot_e}"); logging.error(f"Plotly error: {plot_e}", exc_info=True)
                else: st.warning("No valid OHLC data in period.")
            else: st.warning(f"Missing or non-numeric OHLC columns.")

            st.subheader("📊 Volume Trend")
            if 'volume' in df.columns and pd.notna(df['volume']).any() and pd.api.types.is_numeric_dtype(df['volume']):
                volume_chart = alt.Chart(df_reset.dropna(subset=['volume'])).mark_bar(opacity=0.7).encode(
                     x=alt.X('date', axis=alt.Axis(title='Date')),
                     y=alt.Y('volume', axis=alt.Axis(title='Volume')),
                     tooltip=['date', alt.Tooltip('volume', format=',.0f')]
                 ).properties(title="Trading Volume").interactive()
                st.altair_chart(volume_chart, use_container_width=True)
            else: st.warning("Volume column missing, empty or non-numeric.")

            st.subheader("📉 Daily Returns Distribution")
            if 'daily_return' in df.columns and pd.notna(df['daily_return']).any():
                 returns_chart = alt.Chart(df_reset.dropna(subset=['daily_return'])).mark_bar().encode(
                     alt.X("daily_return", bin=alt.Bin(maxbins=50), title="Daily Return (%)"),
                     alt.Y('count()', title="Frequency"),
                     tooltip=[alt.Tooltip("daily_return", bin=True, title="Return Range (%)"), alt.Tooltip('count()', title="Frequency")]
                 ).properties(title="Distribution of Daily Returns").interactive()
                 st.altair_chart(returns_chart, use_container_width=True)
            else: st.warning("Could not plot daily returns distribution.")

        st.divider()
        st.subheader("⚖️ Positive vs. Negative Return Days")
        if 'daily_return' in df.columns and pd.notna(df['daily_return']).any():
             df_reset['day_type'] = df_reset['daily_return'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
             day_counts = df_reset['day_type'].value_counts().reset_index()
             day_counts.columns = ['type', 'count']

             pie_chart = alt.Chart(day_counts).mark_arc(outerRadius=120).encode(
                 theta=alt.Theta(field="count", type="quantitative"),
                 color=alt.Color(field="type", type="nominal",
                                 scale=alt.Scale(domain=['Positive', 'Negative', 'Neutral'],
                                                 range=['#2ca02c', '#d62728', '#7f7f7f']),
                                 legend=alt.Legend(title="Day Type")),
                 order=alt.Order("count", sort="descending"),
                 tooltip=['type', 'count']
             ).properties(title="Proportion of Trading Day Outcomes")
             st.altair_chart(pie_chart, use_container_width=True)
        else:
             st.warning("Could not create Pie Chart: Daily returns unavailable.")

        st.subheader("🫧 Price vs. Volume Over Time (Bubble Chart)")
        if 'close' in df.columns and 'volume' in df.columns and pd.notna(df['close']).any() and pd.notna(df['volume']).any():
             df_sample = df_reset.dropna(subset=['close', 'volume'])
             if len(df_sample) > 1000:
                 df_sample = df_sample.sample(1000, random_state=42).sort_values('date')

             bubble_chart = alt.Chart(df_sample).mark_circle().encode(
                 x=alt.X('date', axis=alt.Axis(title='Date')),
                 y=alt.Y('close', axis=alt.Axis(title='Closing Price', format='$,.2f')),
                 size=alt.Size('volume', title='Volume', scale=alt.Scale(range=[50, 2000])), # Adjust range for bubble size
                 color=alt.Color('daily_return', scale=alt.Scale(scheme='redblue', domainMid=0), legend=alt.Legend(title="Daily Return (%)")) if 'daily_return' in df_sample else alt.value('grey'),
                 tooltip=[
                     alt.Tooltip('date', format='%Y-%m-%d'),
                     alt.Tooltip('close', format='$,.2f'),
                     alt.Tooltip('volume', format=',.0f'),
                     alt.Tooltip('daily_return', format='.2f', title='Return (%)') if 'daily_return' in df_sample else alt.value(None)
                 ]
             ).properties(
                 title="Price vs. Volume (Bubble Size = Volume, Color = Daily Return)"
             ).interactive()
             st.altair_chart(bubble_chart, use_container_width=True)
        else:
             st.warning("Could not create Bubble Chart: 'close' or 'volume' data unavailable.")


        st.divider()
        st.subheader("Raw Data Sample (Last 10 Rows)")
        st.dataframe(df.tail(10))

if st.session_state.view == "Live Analysis":
    render_live_analysis_page()
elif st.session_state.view == "Historical Analysis":
    render_historical_analysis_page()
else:
    st.error("Invalid view selected."); st.session_state.view = "Live Analysis"; st.rerun()

with st.sidebar:
     st.divider()
     st.info("AI Trading Insights Dashboard")
     st.markdown("Developed using ARDL models & Hugging Face AI.")

