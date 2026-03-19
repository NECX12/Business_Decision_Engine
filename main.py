import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import logging

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSanitizer:
    """Handles data cleaning and validation for the ML pipeline."""
    
    @staticmethod
    def validate_and_clean(df, min_days=30):
        """
        Ensures the business has enough data and fills missing dates.
        Input df must have 'Date' and 'Total Amount' columns.
        """
        # 1. Basic Validation
        if df is None or len(df) < min_days:
            return False, f"Insufficient data: Need at least {min_days} days of history."

        # 2. Standardize Columns for Prophet
        df = df.rename(columns={'Date': 'ds', 'Total Amount': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # 3. Aggregate multiple transactions per day
        df = df.groupby('ds')['y'].sum().reset_index()

        # 4. Fill Missing Dates with 0 (Crucial for time-series stability)
        all_dates = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')
        df = df.set_index('ds').reindex(all_dates).fillna(0).reset_index()
        df.columns = ['ds', 'y']

        # 5. Cap Outliers using IQR (Stabilizes Trend Detection)
        q1 = df['y'].quantile(0.25)
        q3 = df['y'].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + (1.5 * iqr)
        df['y'] = np.where(df['y'] > upper_limit, upper_limit, df['y'])

        return True, df

class RevenueForecaster:
    """Core ML logic for the Smart Business Analytics Dashboard."""
    
    def __init__(self, business_id):
        self.business_id = business_id
        self.model = None

    def train(self, cleaned_df):
        """Trains a Prophet model with Nigerian holiday context."""
        try:
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95  # 95% confidence interval
            )
            # Add Nigerian context
            self.model.add_country_holidays(country_name='NG')
            self.model.fit(cleaned_df)
            logger.info(f"Model trained successfully for business: {self.business_id}")
            return True
        except Exception as e:
            logger.error(f"Training failed for {self.business_id}: {str(e)}")
            return False

    def predict(self, periods=90):
        """Generates forecast and confidence intervals."""
        if not self.model:
            return None
            
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        # Filter for only the future dates
        results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        # Rename for the Django Developer's API requirements
        results = results.rename(columns={
            'ds': 'forecast_date',
            'yhat': 'predicted_revenue',
            'yhat_lower': 'conf_lower',
            'yhat_upper': 'conf_upper'
        })
        
        return results.to_dict(orient='records')

    def get_bde_insight(self, forecast_data):
        """The Business Decision Engine (BDE) Logic."""
        # Simple logic: Compare forecasted average to historical average
        # This can be expanded as you build the BDE module
        avg_forecast = np.mean([x['predicted_revenue'] for x in forecast_data])
        
        insight = {
            "summary": "Stable",
            "message": "Revenue is projected to remain steady over the next period."
        }
        
        if avg_forecast < 0: # Basic safety check
             insight["message"] = "Warning: Model predicts a significant downturn."
             
        return insight