import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import logging
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller

def _preprocess_data(stock_data, sentiment_lags=5, ar_lags=5):
    logging.info("Starting preprocessing pipeline...")

    ardl_df = pd.DataFrame(index=stock_data.index)
    ardl_df['y'] = stock_data['Close'].pct_change() * 100

    logging.info("Calculating and winsorizing returns...")
    returns_not_na = ardl_df['y'].notna()
    if returns_not_na.any():
        original_values = ardl_df.loc[returns_not_na, 'y'].values
        winsorized_values = winsorize(original_values, limits=[0.01, 0.01]).squeeze()
        ardl_df.loc[returns_not_na, 'y'] = winsorized_values
    else:
        logging.warning("No non-NaN returns found to winsorize.")

    logging.info("Returns calculated and winsorized at 1% and 99% levels.")

    adf_result = adfuller(ardl_df['y'].dropna())
    logging.info(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    if adf_result[1] > 0.05:
        logging.warning(f"Returns data p-value ({adf_result[1]:.4f}) is > 0.05. Data may not be stationary.")
    else:
        logging.info("Returns data is stationary (p-value <= 0.05).")

    for i in range(1, sentiment_lags + 1):
        ardl_df[f'sentiment_lag_{i}'] = stock_data['sentiment'].shift(i)

    ardl_df = ardl_df.dropna(subset=['y'])

    if ardl_df.empty:
        logging.error("Preprocessing resulted in an empty DataFrame after dropping initial NaNs.")
        return pd.Series(dtype=np.float64), pd.DataFrame()

    final_returns = ardl_df['y']
    dl_cols = [col for col in ardl_df.columns if 'sentiment_lag' in col]
    final_exog = ardl_df[dl_cols]
    combined_data = pd.concat([final_returns, final_exog], axis=1).dropna()

    if combined_data.empty:
        logging.error("Preprocessing resulted in an empty DataFrame after aligning returns and sentiment lags.")
        return pd.Series(dtype=np.float64), pd.DataFrame()

    final_returns = combined_data['y']
    final_exog = combined_data.drop(columns='y')

    if final_exog.empty:
        final_exog = None

    logging.info(f"Preprocessing complete. Final dataset has {len(final_returns)} observations.")
    return final_returns, final_exog

def fit_ardl_volatility_model(stock_data, recent_sentiment, ar_lags=5, sentiment_lags=5, forecast_horizon=10):
    logging.info("Starting new 2-stage ARDL model fitting and forecasting process.")
    try:
        returns, exog_df = _preprocess_data(
            stock_data=stock_data,
            sentiment_lags=sentiment_lags,
            ar_lags=ar_lags
        )

        if returns.empty:
            logging.error("Preprocessing returned empty data. Cannot fit model.")
            return pd.DataFrame(), pd.Series()

        logging.info(f"Fitting ARDL({ar_lags},{sentiment_lags}) model to mean returns...")

        mean_model = ARIMA(returns, order=(ar_lags, 0, 0), exog=exog_df)
        mean_model_fit = mean_model.fit()

        logging.info("Mean ARDL model fitting complete.")
        print("--- ARDL Mean Model Results ---")
        print(mean_model_fit.summary())

        residuals = mean_model_fit.resid
        fitted_mean_values = mean_model_fit.fittedvalues
        log_sq_residuals = np.log(residuals**2 + 1e-6)
        log_sq_residuals.name = "log_vol"
        vol_data_combined = pd.concat([log_sq_residuals, exog_df.loc[residuals.index]], axis=1).dropna()

        if vol_data_combined.empty:
             logging.error("Volatility modeling data became empty after alignment. Check lags.")
             return pd.DataFrame(), pd.Series()

        final_log_vol = vol_data_combined['log_vol']
        final_vol_exog = vol_data_combined.drop(columns='log_vol')

        if final_vol_exog.empty:
            final_vol_exog = None

        logging.info(f"Fitting ARDL({ar_lags},{sentiment_lags}) model to log-volatility...")
        vol_model = ARIMA(final_log_vol, order=(ar_lags, 0, 0), exog=final_vol_exog)
        vol_model_fit = vol_model.fit()

        logging.info("Volatility ARDL model fitting complete.")
        print("--- ARDL Volatility Model Results ---")
        print(vol_model_fit.summary())

        logging.info(f"Generating mean and volatility forecast for the next {forecast_horizon} days.")
        future_exog_df = None
        if final_vol_exog is not None:
            future_exog_data = {}
            last_sentiments = stock_data['sentiment'].iloc[-sentiment_lags:].tolist()

            for i in range(forecast_horizon):
                current_sentiments = last_sentiments + [recent_sentiment] * (i + 1)
                for j in range(1, sentiment_lags + 1):
                    col_name = f'sentiment_lag_{j}'
                    if col_name not in future_exog_data:
                        future_exog_data[col_name] = []
                    future_exog_data[col_name].append(current_sentiments[len(last_sentiments) + i - j])
            future_exog_df = pd.DataFrame(future_exog_data, columns=final_vol_exog.columns)


        mean_forecast_percent = mean_model_fit.forecast(steps=forecast_horizon, exog=future_exog_df)
        log_vol_forecast = vol_model_fit.forecast(steps=forecast_horizon, exog=future_exog_df)
        predicted_sq_variance = np.exp(log_vol_forecast)
        predicted_variance = predicted_sq_variance.values
        predicted_volatility = np.sqrt(predicted_variance)
        annualized_volatility = predicted_volatility * np.sqrt(252)
        last_actual_price = stock_data['Close'].iloc[-1]
        predicted_prices = []
        current_price = last_actual_price

        for percent_change in mean_forecast_percent:
            current_price *= (1 + percent_change / 100)
            predicted_prices.append(current_price)
        predicted_prices = np.array(predicted_prices)
        price_changes = np.diff(np.insert(predicted_prices, 0, last_actual_price))
        std_dev = (predicted_volatility / 100) * predicted_prices
        lower_band = predicted_prices - 1.96 * std_dev
        upper_band = predicted_prices + 1.96 * std_dev
        last_obs_date = returns.index[-1]
        forecast_dates = pd.to_datetime(last_obs_date) + pd.to_timedelta(np.arange(1, forecast_horizon + 1), 'D')

        forecast_df = pd.DataFrame({
            "Date": forecast_dates.strftime('%Y-%m-%d'),
            "Predicted Price": predicted_prices,
            "Price Change": price_changes,
            "Predicted Return (%)": mean_forecast_percent.values,
            "Lower Band": lower_band,
            "Upper Band": upper_band,
            "Predicted Daily Volatility (%)": predicted_volatility,
            "Predicted Annualized Volatility (%)": annualized_volatility
        })

        forecast_df.to_csv("forecast_output.csv", index=False)
        logging.info("Forecast data saved to forecast_output.csv")
        return forecast_df, fitted_mean_values

    except Exception as e:
        logging.error(f"An error occurred during ARDL modeling: {e}", exc_info=True)
        return pd.DataFrame(), pd.Series()