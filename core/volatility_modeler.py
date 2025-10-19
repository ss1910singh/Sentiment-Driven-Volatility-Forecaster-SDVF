import logging
import pandas as pd
from arch import arch_model

def fit_garch_model_and_forecast(stock_data_with_sentiment: pd.DataFrame, forecast_horizon: int = 5) -> pd.DataFrame:
    logging.info("Starting GARCH model fitting and forecasting process.")
    
    try:
        returns = 100 * stock_data_with_sentiment['Close'].pct_change().dropna()
        exog_sentiment = stock_data_with_sentiment['sentiment'][returns.index]

        if len(returns) < 50: 
            logging.error("Insufficient data to fit GARCH model. Need at least 50 data points.")
            raise ValueError("Not enough historical data for modeling.")

        model = arch_model(
            returns,
            x=exog_sentiment,
            mean='ARX',
            lags=1,
            vol='GARCH',
            p=1,
            q=1,
            dist='StudentsT'
        )

        logging.info("Fitting the ARX-GARCH(1,1) model...")
        results = model.fit(disp='off')
        
        logging.info("Model fitting complete.\n" + str(results.summary()))
        logging.info(f"Generating volatility forecast for the next {forecast_horizon} days.")

        forecast = results.forecast(horizon=forecast_horizon, method='simulation')
        predicted_variance = forecast.variance.iloc[-1]
        predicted_volatility = pd.Series(np.sqrt(predicted_variance.values), name="Predicted Volatility (%)")
        last_date = returns.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
        
        forecast_df = pd.DataFrame({
            'Forecast Date': future_dates,
            'Predicted Daily Volatility (%)': predicted_volatility.values
        })

        forecast_df['Predicted Annualized Volatility (%)'] = forecast_df['Predicted Daily Volatility (%)'] * np.sqrt(252)
        logging.info("Successfully generated volatility forecast.")
        logging.info("Saving forecast output to forecast_output.csv")
        forecast_df.to_csv("forecast_output.csv", index=False)

        return forecast_df

    except Exception as e:
        logging.error(f"An error occurred during GARCH modeling: {e}")
        return pd.DataFrame()