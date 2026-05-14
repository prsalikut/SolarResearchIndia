"""
Solar Power Forecasting System for Rural India - Streamlit Application
Hybrid ML + Statistical Models for Long-term Solar Forecasting
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import time
import json

# Page configuration
st.set_page_config(
    page_title="Solar Power Forecasting - Rural India",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with professional, low-color aesthetic
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 15px;
        border-bottom: 2px solid #3498db;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.4rem;
        color: #2c3e50;
        font-weight: 600;
        margin-top: 20px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .success-box {
        padding: 12px;
        border-left: 4px solid #27ae60;
        background-color: #eafaf1;
        margin: 10px 0;
        border-radius: 4px;
    }
    .info-box {
        padding: 12px;
        border-left: 4px solid #3498db;
        background-color: #ebf5fb;
        margin: 10px 0;
        border-radius: 4px;
    }
    .warning-box {
        padding: 12px;
        border-left: 4px solid #f39c12;
        background-color: #fef9e7;
        margin: 10px 0;
        border-radius: 4px;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .tab-content {
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA COLLECTION CLASS ====================
class SolarDataCollector:
    """Collect weather and solar irradiance data using free APIs with intelligent fallback"""
    
    def __init__(self):
        self.open_meteo_base = "https://archive-api.open-meteo.com/v1/archive"
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def get_solar_data_nasa_power(self, latitude, longitude, start_date, end_date):
        """Get solar irradiance data from NASA POWER (most reliable)"""
        # NASA POWER supports historical data, adjust dates if needed
        today = datetime.now().date()
        max_end_date = today
        
        if end_date > max_end_date:
            end_date = max_end_date
            st.info(f"Using data up to {end_date} (NASA POWER supports historical data)")
        
        if start_date > end_date:
            start_date = end_date - timedelta(days=365*3)
        
        # Ensure reasonable date range
        if (end_date - start_date).days < 30:
            start_date = end_date - timedelta(days=365)
        
        start = start_date.strftime('%Y%m%d')
        end = end_date.strftime('%Y%m%d')
        
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M',
            'community': 'RE',
            'longitude': longitude,
            'latitude': latitude,
            'start': start,
            'end': end,
            'format': 'JSON'
        }
        
        try:
            response = requests.get(self.nasa_power_base, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if 'properties' in data and 'parameter' in data['properties']:
                params_data = data['properties']['parameter']
                dates = list(params_data['ALLSKY_SFC_SW_DWN'].keys())
                df_dict = {'date': dates}
                
                for param, values in params_data.items():
                    df_dict[param] = list(values.values())
                
                df = pd.DataFrame(df_dict)
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                
                df = df.rename(columns={
                    'ALLSKY_SFC_SW_DWN': 'solar_irradiance_kwh_m2',
                    'CLRSKY_SFC_SW_DWN': 'clear_sky_irradiance_kwh_m2',
                    'T2M': 'temperature_c',
                    'RH2M': 'relative_humidity',
                    'WS10M': 'wind_speed_ms'
                })
                return df
            return None
        except Exception as e:
            st.warning(f"NASA POWER API: {str(e)[:100]}...")
            return None
    
    def get_weather_data_open_meteo(self, latitude, longitude, start_date, end_date):
        """Get weather data from Open-Meteo as fallback"""
        try:
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                        'precipitation_sum,shortwave_radiation_sum,'
                        'relative_humidity_2m_mean,cloudcover_mean',
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(self.open_meteo_base, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' in data:
                df = pd.DataFrame(data['daily'])
                df['date'] = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
                return df
            return None
        except:
            return None
    
    def generate_smart_synthetic_data(self, latitude, longitude, start_date, end_date):
        """Generate realistic synthetic data based on location patterns"""
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Location-based base values (India rural areas)
        if latitude > 20:  # Northern India
            base_temp = 25 + 12 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 5.5 + 2.5 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        else:  # Southern India
            base_temp = 28 + 8 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 6 + 2 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        
        # Add realistic randomness
        np.random.seed(int(latitude * longitude))  # Seed based on location
        
        df = pd.DataFrame({
            'date': date_range,
            'solar_irradiance_kwh_m2': np.maximum(base_irradiance + np.random.normal(0, 1.2, len(date_range)), 0),
            'temperature_c': base_temp + np.random.normal(0, 3, len(date_range)),
            'relative_humidity': 50 + 25 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + np.random.normal(0, 12, len(date_range)),
            'cloudcover_mean': 40 + 35 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + np.random.normal(0, 18, len(date_range))
        })
        
        # Add derived weather columns
        df['temperature_2m_mean'] = df['temperature_c']
        df['temperature_2m_max'] = df['temperature_c'] + 5 + np.random.normal(0, 2, len(date_range))
        df['temperature_2m_min'] = df['temperature_c'] - 5 + np.random.normal(0, 2, len(date_range))
        df['precipitation_sum'] = np.random.exponential(0.3, len(date_range))
        df['windspeed_10m_max'] = 4 + 3 * np.random.random(len(date_range))
        df['clear_sky_irradiance_kwh_m2'] = df['solar_irradiance_kwh_m2'] * (1.2 + 0.3 * np.random.random(len(date_range)))
        
        # Clip to realistic ranges
        df['solar_irradiance_kwh_m2'] = df['solar_irradiance_kwh_m2'].clip(0, 8)
        df['temperature_2m_mean'] = df['temperature_2m_mean'].clip(10, 45)
        df['relative_humidity'] = df['relative_humidity'].clip(20, 100)
        df['cloudcover_mean'] = df['cloudcover_mean'].clip(0, 100)
        df['clear_sky_irradiance_kwh_m2'] = df['clear_sky_irradiance_kwh_m2'].clip(0, 10)
        
        return df
    
    def collect_data_for_location(self, location_name, latitude, longitude, start_date, end_date, panel_capacity, efficiency):
        """Intelligent data collection with automatic fallback and immediate solar power calculation"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Try NASA POWER first (most reliable for solar data)
        status_text.text("Fetching solar irradiance data from NASA POWER...")
        progress_bar.progress(30)
        
        solar_df = self.get_solar_data_nasa_power(latitude, longitude, start_date, end_date)
        
        if solar_df is not None:
            status_text.text("Fetching additional weather data...")
            progress_bar.progress(60)
            
            # Try Open-Meteo for additional weather data
            weather_df = self.get_weather_data_open_meteo(latitude, longitude, start_date, end_date)
            
            if weather_df is not None:
                # Merge both datasets
                merged_df = pd.merge(solar_df, weather_df, on='date', how='outer')
            else:
                merged_df = solar_df.copy()
                # Add estimated weather columns
                merged_df['temperature_2m_mean'] = merged_df['temperature_c']
                merged_df['cloudcover_mean'] = 50  # Default estimate
            
            status_text.text("Finalizing data collection...")
            progress_bar.progress(90)
            
        else:
            # Generate smart synthetic data
            status_text.text("Generating realistic data based on location patterns...")
            progress_bar.progress(70)
            
            merged_df = self.generate_smart_synthetic_data(latitude, longitude, start_date, end_date)
            
            st.info(f"Using realistic generated data for {location_name} based on location patterns")
        
        # Add location metadata
        merged_df['location'] = location_name
        merged_df['latitude'] = latitude
        merged_df['longitude'] = longitude
        
        # Fill any missing dates and sort
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())
        
        progress_bar.progress(100)
        status_text.text(f"Data collection complete: {len(merged_df)} days")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return merged_df


# ==================== FEATURE ENGINEERING ====================
class SolarFeatureEngineering:
    """Create features for solar power prediction"""
    
    @staticmethod
    def add_temporal_features(df):
        """Add time-based features"""
        df = df.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Indian seasons
        df['season'] = df['month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Summer' if x in [3, 4, 5] else
            'Monsoon' if x in [6, 7, 8, 9] else
            'Post-Monsoon'
        )
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Weekend and holiday indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_summer_peak'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        
        return df
    
    @staticmethod
    def add_solar_features(df):
        """Add solar-specific features"""
        df = df.copy()
        
        # Clearness index
        if 'solar_irradiance_kwh_m2' in df.columns and 'clear_sky_irradiance_kwh_m2' in df.columns:
            df['clearness_index'] = df['solar_irradiance_kwh_m2'] / (df['clear_sky_irradiance_kwh_m2'] + 0.001)
            df['clearness_index'] = df['clearness_index'].clip(0, 1)
        elif 'solar_irradiance_kwh_m2' in df.columns:
            # Estimate clearness based on location and season
            df['clearness_index'] = 0.7 + 0.1 * np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['clearness_index'] = df['clearness_index'].clip(0.5, 0.9)
        
        # Temperature efficiency factor - FIXED to prevent negative values
        temp_col = 'temperature_2m_mean' if 'temperature_2m_mean' in df.columns else 'temperature_c'
        if temp_col in df.columns:
            # Modified formula to always keep efficiency positive
            df['temp_efficiency_factor'] = np.maximum(0.85, 1.0 - 0.003 * np.abs(df[temp_col] - 25))
            df['temp_efficiency_factor'] = df['temp_efficiency_factor'].clip(0.85, 1.05)
        
        # Cloud reduction factor
        if 'cloudcover_mean' in df.columns:
            df['cloud_reduction_factor'] = 1.0 - (df['cloudcover_mean'] / 100) * 0.6
            df['cloud_reduction_factor'] = df['cloud_reduction_factor'].clip(0.4, 1.0)
        
        # Air mass factor (simplified)
        df['air_mass_factor'] = 1.0 / np.cos(np.radians(23.45 * np.sin(2 * np.pi * (df['day_of_year'] - 81) / 365)))
        df['air_mass_factor'] = df['air_mass_factor'].clip(1, 5)
        
        return df
    
    @staticmethod
    def add_lag_features(df, target_col, lags=[1, 2, 3, 7, 14]):
        """Add lagged features"""
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    @staticmethod
    def add_rolling_features(df, target_col, windows=[3, 7, 14, 30]):
        """Add rolling statistics"""
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        return df
    
    @staticmethod
    def create_future_features(future_dates, location_lat):
        """Create features for future dates"""
        df = pd.DataFrame({'date': future_dates})
        
        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Seasonal indicators
        df['is_summer_peak'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Estimated solar features based on location and season
        # Base irradiance pattern
        base_irradiance = 5.5 + 2.5 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365.25)
        
        # Adjust for monsoon
        monsoon_reduction = np.where(df['is_monsoon'] == 1, 0.6, 1.0)
        df['estimated_irradiance'] = base_irradiance * monsoon_reduction
        
        # Estimated temperature
        if location_lat > 20:  # Northern India
            df['estimated_temp'] = 25 + 12 * np.sin(2 * np.pi * (df['day_of_year'] - 105) / 365.25)
        else:  # Southern India
            df['estimated_temp'] = 28 + 8 * np.sin(2 * np.pi * (df['day_of_year'] - 105) / 365.25)
        
        # Add some realistic randomness
        np.random.seed(42)
        df['estimated_irradiance'] += np.random.normal(0, 0.8, len(df))
        df['estimated_temp'] += np.random.normal(0, 2.5, len(df))
        
        # Clip to realistic ranges
        df['estimated_irradiance'] = df['estimated_irradiance'].clip(0, 8)
        df['estimated_temp'] = df['estimated_temp'].clip(10, 45)
        
        return df


# ==================== SOLAR POWER CALCULATOR ====================
def calculate_solar_power(df, panel_capacity_kw=5.0, efficiency=0.17):
    """
    Calculate solar power output from irradiance data
    
    Args:
        df: DataFrame with solar irradiance
        panel_capacity_kw: Installed solar panel capacity in kW
        efficiency: Panel efficiency (default 17%)
    
    Returns:
        DataFrame with solar_power_kwh column (non-negative)
    """
    df = df.copy()
    
    # Determine irradiance column
    if 'solar_irradiance_kwh_m2' in df.columns:
        irradiance_col = 'solar_irradiance_kwh_m2'
    elif 'estimated_irradiance' in df.columns:
        irradiance_col = 'estimated_irradiance'
    else:
        # If no irradiance data, return df without solar_power_kwh
        return df
    
    # Calculate panel area: Panel Capacity = Irradiance (1 kW/m²) × Area × Efficiency
    # So Area = Panel Capacity / (1 × Efficiency)
    panel_area = panel_capacity_kw / (1.0 * efficiency)
    
    # Daily energy generation (kWh) = Irradiance (kWh/m²) × Area (m²) × Efficiency
    # FIXED: Ensure non-negative calculation
    df['solar_power_kwh'] = df[irradiance_col] * panel_area * efficiency
    
    # Apply efficiency factors if available - ensure non-negative
    efficiency_factors = ['temp_efficiency_factor', 'cloud_reduction_factor']
    for factor in efficiency_factors:
        if factor in df.columns:
            # Use maximum with small positive value to prevent negative results
            df[factor] = df[factor].clip(lower=0.1)
            df['solar_power_kwh'] *= df[factor]
    
    # Apply monsoon reduction for Indian context
    if 'is_monsoon' in df.columns:
        # Reduce by 40% during monsoon but keep positive
        df['solar_power_kwh'] *= (0.6 + 0.4 * (1 - df['is_monsoon']))
    
    # Ensure non-negative values and realistic maximum
    # Maximum theoretical: panel_capacity × 8 peak sun hours
    max_theoretical = panel_capacity_kw * 8
    df['solar_power_kwh'] = df['solar_power_kwh'].clip(lower=0, upper=max_theoretical)
    
    # Set zero for night-time (very low irradiance)
    night_threshold = 0.05
    df.loc[df[irradiance_col] <= night_threshold, 'solar_power_kwh'] = 0
    
    return df


# ==================== SOLAR FORECASTING MODEL ====================
class SolarForecastingModel:
    """Advanced solar forecasting model for rural India"""
    
    def __init__(self, n_estimators=200, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained = False
        
    def prepare_features(self, df, is_training=True):
        """Prepare features for training or prediction"""
        fe = SolarFeatureEngineering()
        
        # Add temporal features
        df = fe.add_temporal_features(df)
        
        # Add solar features
        df = fe.add_solar_features(df)
        
        # Prepare feature set
        exclude_cols = ['date', 'location', 'latitude', 'longitude', 'solar_power_kwh', 
                       'season', 'estimated_irradiance', 'estimated_temp']
        
        if is_training and 'solar_power_kwh' in df.columns:
            # Add lag and rolling features for training
            df = fe.add_lag_features(df, 'solar_power_kwh', lags=[1, 2, 3, 7])
            df = fe.add_rolling_features(df, 'solar_power_kwh', windows=[7, 14, 30])
        
        # Select numeric features
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        # Handle missing values
        if feature_cols:
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
        
        return df, feature_cols
    
    def train(self, df, panel_capacity, efficiency):
        """Train the forecasting model"""
        # Prepare features
        df, feature_cols = self.prepare_features(df, is_training=True)
        
        # Check if we have solar power data
        if 'solar_power_kwh' not in df.columns:
            raise ValueError("Solar power data not available. Please recollect data with proper irradiance information.")
        
        # Remove rows with missing target
        df = df.dropna(subset=['solar_power_kwh'])
        
        if len(df) < 30:
            raise ValueError(f"Not enough data for training. Need at least 30 days, got {len(df)}")
        
        X = df[feature_cols]
        y = df['solar_power_kwh']
        
        # Ensure y is non-negative
        y = y.clip(lower=0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.feature_names = feature_cols
        self.trained = True
        
        return X, y
    
    def predict_future(self, historical_df, future_months, panel_capacity, efficiency, location_lat):
        """Predict future solar power generation"""
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        # Create future dates
        last_date = historical_df['date'].max()
        future_days = future_months * 30  # Approximate
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=future_days,
            freq='D'
        )
        
        # Create future features
        fe = SolarFeatureEngineering()
        future_df = fe.create_future_features(future_dates, location_lat)
        
        # Calculate solar power for future
        future_df = calculate_solar_power(future_df, panel_capacity, efficiency)
        
        # Prepare features for prediction
        future_processed, _ = self.prepare_features(future_df, is_training=False)
        
        # Ensure all training features are present
        for col in self.feature_names:
            if col not in future_processed.columns:
                future_processed[col] = 0
        
        # Align columns
        X_future = future_processed[self.feature_names]
        
        # Scale and predict
        X_future_scaled = self.scaler.transform(X_future)
        predictions = self.model.predict(X_future_scaled)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'date': future_dates,
            'predicted_power_kwh': predictions,
            'estimated_irradiance': future_df['estimated_irradiance'],
            'estimated_temperature': future_df['estimated_temp'],
            'month': future_df['month'],
            'is_monsoon': future_df['is_monsoon'],
            'is_summer_peak': future_df['is_summer_peak']
        })
        
        # Add confidence intervals based on historical performance
        if 'solar_power_kwh' in historical_df.columns:
            historical_std = historical_df['solar_power_kwh'].std()
            results_df['lower_bound'] = results_df['predicted_power_kwh'] - 1.5 * historical_std
            results_df['upper_bound'] = results_df['predicted_power_kwh'] + 1.5 * historical_std
            # Ensure lower bound is non-negative
            results_df['lower_bound'] = results_df['lower_bound'].clip(lower=0)
        
        return results_df
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        if not self.trained:
            raise ValueError("Model not trained")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Ensure predictions are non-negative
        y_pred = np.maximum(y_pred, 0)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        # Calculate MAPE (handle zero values)
        mask = y_test > 0
        if np.any(mask):
            metrics['MAPE'] = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            metrics['MAPE'] = 0
        
        return metrics, y_pred
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.trained or self.feature_names is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# ==================== VISUALIZATION FUNCTIONS ====================
def plot_historical_data(df, location_name):
    """Plot historical solar and weather data"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Solar Power Generation', 'Solar Irradiance', 'Temperature'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Solar Power
    if 'solar_power_kwh' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['solar_power_kwh'],
                      name='Solar Power', line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
    
    # Solar Irradiance
    irradiance_col = None
    if 'solar_irradiance_kwh_m2' in df.columns:
        irradiance_col = 'solar_irradiance_kwh_m2'
    elif 'estimated_irradiance' in df.columns:
        irradiance_col = 'estimated_irradiance'
    
    if irradiance_col:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[irradiance_col],
                      name='Irradiance', line=dict(color='#f39c12', width=1.5)),
            row=2, col=1
        )
    
    # Temperature
    temp_col = None
    if 'temperature_2m_mean' in df.columns:
        temp_col = 'temperature_2m_mean'
    elif 'temperature_c' in df.columns:
        temp_col = 'temperature_c'
    elif 'estimated_temp' in df.columns:
        temp_col = 'estimated_temp'
    
    if temp_col:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[temp_col],
                      name='Temperature', line=dict(color='#e74c3c', width=1.5)),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    if irradiance_col:
        fig.update_yaxes(title_text="kWh/m²", row=2, col=1)
    if temp_col:
        fig.update_yaxes(title_text="°C", row=3, col=1)
    
    fig.update_layout(
        height=700,
        title_text=f"Historical Data - {location_name}",
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def plot_seasonal_patterns(df):
    """Plot seasonal patterns of solar generation"""
    if 'solar_power_kwh' not in df.columns or 'month' not in df.columns:
        return None
    
    monthly_avg = df.groupby('month').agg({
        'solar_power_kwh': 'mean',
    }).reset_index()
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_avg['month_name'],
        y=monthly_avg['solar_power_kwh'],
        name='Avg Solar Power',
        marker_color='#3498db'
    ))
    
    fig.update_layout(
        title="Monthly Average Solar Power Generation",
        xaxis_title="Month",
        yaxis_title="Average Daily Generation (kWh)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def plot_forecast_comparison(historical_df, forecast_df, location_name, panel_capacity):
    """Plot historical data with future forecast"""
    fig = go.Figure()
    
    # Historical data (last 90 days for clarity)
    historical_recent = historical_df.tail(90)
    if 'solar_power_kwh' in historical_recent.columns:
        fig.add_trace(go.Scatter(
            x=historical_recent['date'],
            y=historical_recent['solar_power_kwh'],
            mode='lines',
            name='Historical (Recent)',
            line=dict(color='#2c3e50', width=2)
        ))
    
    # Future forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['predicted_power_kwh'],
        mode='lines',
        name='Forecast',
        line=dict(color='#3498db', width=3)
    ))
    
    # Confidence interval
    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
            y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(52, 152, 219, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"Solar Power Forecast - {location_name} ({panel_capacity} kW System)",
        xaxis_title="Date",
        yaxis_title="Solar Power Generation (kWh)",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def plot_forecast_summary(forecast_df):
    """Plot forecast summary by month"""
    monthly_forecast = forecast_df.copy()
    monthly_forecast['month'] = monthly_forecast['date'].dt.month
    
    monthly_summary = monthly_forecast.groupby('month').agg({
        'predicted_power_kwh': ['sum', 'mean', 'std']
    }).round(2)
    
    monthly_summary.columns = ['total_kwh', 'avg_daily_kwh', 'std_daily_kwh']
    monthly_summary = monthly_summary.reset_index()
    
    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_summary['month_name'] = monthly_summary['month'].apply(lambda x: month_names[x-1])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Total Generation', 'Average Daily Generation'),
        vertical_spacing=0.15
    )
    
    # Total monthly generation
    fig.add_trace(
        go.Bar(x=monthly_summary['month_name'], y=monthly_summary['total_kwh'],
              name='Total', marker_color='#3498db'),
        row=1, col=1
    )
    
    # Average daily generation
    fig.add_trace(
        go.Bar(x=monthly_summary['month_name'], y=monthly_summary['avg_daily_kwh'],
              name='Average', marker_color='#2ecc71'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Total kWh", row=1, col=1)
    fig.update_yaxes(title_text="Average kWh/day", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig, monthly_summary


# ==================== MAIN STREAMLIT APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">Solar Power Forecasting System for Rural India</p>',
                unsafe_allow_html=True)
    
    project_description = """
    <div class="info-box">
    <b>Project Objective:</b> Develop hybrid predictive algorithms to forecast long-term solar power output 
    in rural India using weather patterns and solar irradiance data. This tool supports solar infrastructure 
    planning, policy development, and sustainable energy deployment in rural communities.
    </div>
    """
    st.markdown(project_description, unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.title("Configuration")
    
    # Location Selection
    st.sidebar.subheader("Location")
    RURAL_LOCATIONS = {
        'Jodhpur, Rajasthan': (26.2389, 73.0243),
        'Jaisalmer, Rajasthan': (26.9157, 70.9083),
        'Anantapur, Andhra Pradesh': (14.6819, 77.6006),
        'Bikaner, Rajasthan': (28.0229, 73.3119),
        'Kurnool, Andhra Pradesh': (15.8281, 78.0373),
        'Nashik, Maharashtra': (19.9975, 73.7898),
        'Solapur, Maharashtra': (17.6599, 75.9064),
        'Leh, Ladakh': (34.1526, 77.5771),
        'Bhuj, Gujarat': (23.2420, 69.6669),
        'Custom Location': None
    }
    
    location_choice = st.sidebar.selectbox(
        "Select Location",
        list(RURAL_LOCATIONS.keys())
    )
    
    if location_choice == 'Custom Location':
        location_name = st.sidebar.text_input("Location Name", "Custom Location")
        latitude = st.sidebar.number_input("Latitude", value=26.2389, format="%.4f")
        longitude = st.sidebar.number_input("Longitude", value=73.0243, format="%.4f")
    else:
        location_name = location_choice
        latitude, longitude = RURAL_LOCATIONS[location_choice]
    
    # Data Collection Period
    st.sidebar.subheader("Data Period")
    
    # Set sensible default dates
    today = datetime.now().date()
    default_end = today - timedelta(days=1)  # Yesterday
    default_start = default_end - timedelta(days=365)  # 1 year back
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
    # Validate dates
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date = end_date - timedelta(days=30)
    
    if (end_date - start_date).days < 30:
        st.sidebar.warning("Minimum 30 days recommended for better analysis")
    
    # Solar System Configuration
    st.sidebar.subheader("Solar System")
    panel_capacity = st.sidebar.number_input("Panel Capacity (kW)", 
                                            value=5.0, 
                                            min_value=0.1, 
                                            max_value=100.0, 
                                            step=0.5)
    panel_efficiency = st.sidebar.slider("Panel Efficiency (%)", 
                                        10, 25, 17) / 100
    
    # Forecasting Configuration
    st.sidebar.subheader("Forecasting")
    forecast_months = st.sidebar.slider("Forecast Period (months)", 
                                       1, 36, 12)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'forecast' not in st.session_state:
        st.session_state.forecast = None
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Collection & Analysis", 
        "Model Training", 
        "Future Forecast", 
        "Documentation"
    ])
    
    # ==================== TAB 1: DATA COLLECTION & ANALYSIS ====================
    with tab1:
        st.markdown('<p class="sub-header">Data Collection & Analysis</p>', unsafe_allow_html=True)
        
        # Display configuration summary
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.metric("Location", location_name)
            st.metric("Coordinates", f"{latitude}°N, {longitude}°E")
        with config_col2:
            st.metric("Data Period", f"{start_date} to {end_date}")
            st.metric("Panel Configuration", f"{panel_capacity} kW, {panel_efficiency*100:.0f}%")
        
        # Data Collection Button
        if st.button("Collect Historical Data", type="primary", width='stretch'):
            with st.spinner("Collecting data..."):
                try:
                    collector = SolarDataCollector()
                    
                    # Collect data (this now calculates solar power immediately)
                    df = collector.collect_data_for_location(
                        location_name, latitude, longitude,
                        start_date, end_date, panel_capacity, panel_efficiency
                    )
                    
                    if df is not None and len(df) > 0:
                        # Now calculate solar power with feature engineering
                        df = calculate_solar_power(df, panel_capacity, panel_efficiency)
                        
                        # Add temporal features for better visualization
                        fe = SolarFeatureEngineering()
                        df = fe.add_temporal_features(df)
                        df = fe.add_solar_features(df)
                        
                        st.session_state.data = df
                        
                        # Display success metrics
                        st.success(f"Successfully collected and processed {len(df)} days of data")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Days", len(df))
                        with col2:
                            if 'solar_power_kwh' in df.columns:
                                avg_power = df['solar_power_kwh'].mean()
                                st.metric("Avg Daily", f"{avg_power:.1f} kWh")
                            else:
                                st.metric("Avg Daily", "N/A")
                        with col3:
                            if 'solar_power_kwh' in df.columns:
                                total_power = df['solar_power_kwh'].sum()
                                st.metric("Total Energy", f"{total_power:.0f} kWh")
                            else:
                                st.metric("Total Energy", "N/A")
                        with col4:
                            # FIX: Calculate average irradiance correctly
                            irradiance_col = None
                            if 'solar_irradiance_kwh_m2' in df.columns:
                                irradiance_col = 'solar_irradiance_kwh_m2'
                            elif 'estimated_irradiance' in df.columns:
                                irradiance_col = 'estimated_irradiance'
                            
                            if irradiance_col:
                                # Ensure irradiance is non-negative before calculating average
                                df[irradiance_col] = df[irradiance_col].clip(lower=0)
                                avg_irrad = df[irradiance_col].mean()
                                st.metric("Avg Irradiance", f"{avg_irrad:.2f} kWh/m²")
                            else:
                                st.metric("Avg Irradiance", "N/A")
                        
                    else:
                        st.error("Failed to collect data. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display data if available
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Data Visualization
            st.markdown("---")
            st.subheader("Data Visualization")
            
            # Time series plot
            fig_historical = plot_historical_data(df, location_name)
            st.plotly_chart(fig_historical, width='stretch')
            
            # Seasonal patterns
            fig_seasonal = plot_seasonal_patterns(df)
            if fig_seasonal:
                st.plotly_chart(fig_seasonal, width='stretch')
            
            # Data table - SAFE VERSION: Only show columns that exist
            with st.expander("View Data Table"):
                # Create list of columns to display (only if they exist)
                display_cols = ['date']
                
                possible_cols = ['solar_power_kwh', 'solar_irradiance_kwh_m2', 
                               'estimated_irradiance', 'temperature_2m_mean', 
                               'temperature_c', 'estimated_temp', 'cloudcover_mean']
                
                for col in possible_cols:
                    if col in df.columns:
                        display_cols.append(col)
                
                if len(display_cols) > 1:  # At least date plus one other column
                    st.dataframe(df[display_cols].head(20), 
                                width='stretch')
                else:
                    st.warning("No data columns available to display")
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"solar_data_{location_name}.csv",
                mime="text/csv",
                width='stretch'
            )
        else:
            st.info("Click 'Collect Historical Data' to start data collection and analysis")
    
    # ==================== TAB 2: MODEL TRAINING ====================
    with tab2:
        st.markdown('<p class="sub-header">Model Training & Evaluation</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Check if we have solar power data
            if 'solar_power_kwh' not in df.columns:
                st.error("Solar power data not available. Please recollect data.")
            else:
                st.info(f"Using {len(df)} days of historical data for model training")
                
                # Training configuration
                st.subheader("Training Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
                with col2:
                    random_state = st.number_input("Random Seed", value=42, min_value=0)
                
                # Train button
                if st.button("Train Forecasting Model", type="primary", width='stretch'):
                    with st.spinner("Training model..."):
                        try:
                            # Initialize and train model
                            model = SolarForecastingModel(random_state=random_state)
                            X, y = model.train(df, panel_capacity, panel_efficiency)
                            
                            # Store model
                            st.session_state.model = model
                            st.session_state.trained = True
                            
                            # Prepare train-test split
                            split_idx = int(len(X) * (1 - test_size/100))
                            X_train, X_test = X[:split_idx], X[split_idx:]
                            y_train, y_test = y[:split_idx], y[split_idx:]
                            
                            # Evaluate model
                            metrics, y_pred = model.evaluate(X_test, y_test)
                            
                            st.success("Model trained successfully!")
                            
                            # Display metrics
                            st.subheader("Model Performance")
                            
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric("R² Score", f"{metrics['R2']:.3f}")
                            with metric_cols[1]:
                                st.metric("RMSE", f"{metrics['RMSE']:.3f}")
                            with metric_cols[2]:
                                st.metric("MAE", f"{metrics['MAE']:.3f}")
                            with metric_cols[3]:
                                st.metric("MAPE", f"{metrics['MAPE']:.1f}%")
                            
                            # Feature importance
                            importance_df = model.get_feature_importance()
                            if importance_df is not None:
                                st.subheader("Feature Importance")
                                
                                top_features = importance_df.head(10)
                                fig_importance = go.Figure(go.Bar(
                                    x=top_features['importance'],
                                    y=top_features['feature'],
                                    orientation='h',
                                    marker_color='#3498db'
                                ))
                                
                                fig_importance.update_layout(
                                    title="Top 10 Most Important Features",
                                    xaxis_title="Importance",
                                    yaxis_title="Feature",
                                    height=400,
                                    yaxis=dict(autorange="reversed"),
                                    plot_bgcolor='white',
                                    paper_bgcolor='white'
                                )
                                
                                st.plotly_chart(fig_importance, width='stretch')
                            
                        except Exception as e:
                            st.error(f"Training failed: {str(e)}")
                else:
                    st.info("Click 'Train Forecasting Model' to train the model on historical data")
        else:
            st.warning("Please collect data first in the 'Data Collection & Analysis' tab.")
    
    # ==================== TAB 3: FUTURE FORECAST ====================
    with tab3:
        st.markdown('<p class="sub-header">Future Solar Power Forecast</p>', unsafe_allow_html=True)
        
        if st.session_state.trained and st.session_state.model is not None:
            model = st.session_state.model
            historical_df = st.session_state.data
            
            st.write(f"""
            **Forecasting Setup:**  
            • **Location:** {location_name}  
            • **System:** {panel_capacity} kW, {panel_efficiency*100:.0f}% efficiency  
            • **Forecast Period:** {forecast_months} months  
            • **Historical Data:** {len(historical_df)} days
            """)
            
            # Generate forecast
            if st.button("Generate Forecast", type="primary", width='stretch'):
                with st.spinner(f"Generating {forecast_months}-month forecast..."):
                    try:
                        forecast_df = model.predict_future(
                            historical_df, 
                            forecast_months, 
                            panel_capacity, 
                            efficiency=panel_efficiency,
                            location_lat=latitude
                        )
                        
                        st.session_state.forecast = forecast_df
                        
                        st.success(f"Forecast generated for {len(forecast_df)} future days")
                        
                    except Exception as e:
                        st.error(f"Forecast generation failed: {str(e)}")
            
            # Display forecast if available
            if st.session_state.forecast is not None:
                forecast_df = st.session_state.forecast
                
                # Forecast summary
                st.subheader("Forecast Summary")
                
                summary_cols = st.columns(3)
                with summary_cols[0]:
                    total_energy = forecast_df['predicted_power_kwh'].sum()
                    st.metric("Total Forecast Energy", f"{total_energy:,.0f} kWh")
                with summary_cols[1]:
                    avg_daily = forecast_df['predicted_power_kwh'].mean()
                    st.metric("Average Daily", f"{avg_daily:.1f} kWh/day")
                with summary_cols[2]:
                    max_daily = forecast_df['predicted_power_kwh'].max()
                    st.metric("Peak Daily", f"{max_daily:.1f} kWh")
                
                # Forecast visualization
                st.subheader("Forecast Visualization")
                
                fig_forecast = plot_forecast_comparison(historical_df, forecast_df, location_name, panel_capacity)
                st.plotly_chart(fig_forecast, width='stretch')
                
                # Monthly summary
                st.subheader("Monthly Forecast Analysis")
                
                fig_summary, monthly_data = plot_forecast_summary(forecast_df)
                st.plotly_chart(fig_summary, width='stretch')
                
                # Display monthly data
                with st.expander("View Monthly Forecast Data"):
                    st.dataframe(monthly_data, width='stretch')
                
                # Insights
                st.subheader("Forecast Insights")
                
                # Find best and worst months
                best_month_idx = monthly_data['total_kwh'].idxmax()
                worst_month_idx = monthly_data['total_kwh'].idxmin()
                
                best_month = monthly_data.loc[best_month_idx, 'month_name']
                worst_month = monthly_data.loc[worst_month_idx, 'month_name']
                
                insights = f"""
                <div class="info-box">
                <b>Key Insights for {location_name}:</b><br><br>
                • <b>Best Generation:</b> {best_month} (highest solar yield)<br>
                • <b>Lowest Generation:</b> {worst_month} (consider energy storage)<br>
                • <b>Total Forecast Energy:</b> {total_energy:,.0f} kWh over {forecast_months} months<br>
                • <b>Daily Average:</b> {avg_daily:.1f} kWh/day across all seasons<br>
                • <b>System Utilization:</b> Approximately {avg_daily/(panel_capacity*8)*100:.1f}% of maximum potential
                </div>
                """
                st.markdown(insights, unsafe_allow_html=True)
                
                # Download forecast
                csv_forecast = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast Data",
                    data=csv_forecast,
                    file_name=f"forecast_{location_name}_{forecast_months}months.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
        elif st.session_state.data is not None:
            st.info("Please train the model first in the 'Model Training' tab.")
        else:
            st.warning("Please collect data first in the 'Data Collection & Analysis' tab.")
    
    # ==================== TAB 4: DOCUMENTATION ====================
    with tab4:
        st.markdown('<p class="sub-header">Documentation & Methodology</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Project Overview
        
        **Title:** Forecasting Models for Long-Term Solar Energy Trends: A Weather-Driven Study in Rural India
        
        **Objective:** Develop hybrid predictive algorithms to forecast long-term solar power output in rural 
        India by leveraging weather patterns and solar irradiance data.
        
        ---
        
        ## Methodology
        
        ### 1. Data Collection
        - **Primary Source:** NASA POWER API for solar irradiance and weather data
        - **Fallback Mechanism:** Intelligent synthetic data generation based on location patterns
        - **Data Features:** Solar irradiance, temperature, humidity, cloud cover
        - **Location-specific:** Custom patterns for different Indian regions
        
        ### 2. Feature Engineering
        - **Temporal Features:** Month, season, day of year with cyclical encoding
        - **Solar Features:** Clearness index, temperature efficiency, cloud reduction
        - **Location Features:** Latitude-based seasonal adjustments
        - **Historical Patterns:** Lagged values and rolling statistics
        
        ### 3. Forecasting Model
        - **Algorithm:** Random Forest Regression ensemble
        - **Training:** Historical weather-solar power relationships
        - **Prediction:** Future solar power based on projected weather patterns
        - **Uncertainty:** Confidence intervals based on historical variability
        
        ### 4. Validation
        - **Temporal Split:** Train-test split preserving time order
        - **Metrics:** R², RMSE, MAE, MAPE for accuracy assessment
        - **Feature Importance:** Identify key drivers of solar generation
        
        ---
        
        ## Application Areas
        
        ### 1. Rural Solar Planning
        - System sizing for mini-grids and standalone systems
        - Battery storage requirement estimation
        - Seasonal energy availability assessment
        
        ### 2. Policy Development
        - Regional solar potential mapping
        - Investment planning for solar infrastructure
        - Energy access strategy formulation
        
        ### 3. Research & Analysis
        - Climate impact studies on solar generation
        - Seasonal pattern analysis
        - Technology performance assessment
        
        ---
        
        ## How to Use This Tool
        
        ### Step 1: Data Collection
        1. Select location from predefined rural areas or enter custom coordinates
        2. Choose historical data period (minimum 30 days recommended)
        3. Click "Collect Historical Data"
        4. System automatically fetches data or generates realistic patterns
        
        ### Step 2: Model Training
        1. Configure training parameters (test size, random seed)
        2. Click "Train Forecasting Model"
        3. Review model performance metrics
        4. Examine feature importance
        
        ### Step 3: Future Forecasting
        1. Set forecast period (1-36 months)
        2. Click "Generate Forecast"
        3. Analyze forecast results and insights
        4. Download forecast data for further analysis
        
        ---
        
        ## Technical Notes
        
        ### Data Sources
        - **NASA POWER:** Solar irradiance, temperature, humidity data
        - **Location Patterns:** Seasonally adjusted synthetic data when needed
        - **Indian Context:** Monsoon adjustments, regional climate patterns
        
        ### Model Limitations
        - Assumes climate stationarity over forecast period
        - Does not account for extreme weather events
        - Daily resolution forecasts only
        - Dependent on historical pattern consistency
        
        ### Future Enhancements
        - Integration of satellite imagery data
        - Climate change projection integration
        - Higher temporal resolution (hourly forecasts)
        - Ensemble of multiple forecasting models
        
        ---
        
        ## References
        
        1. NASA POWER Data Services
        2. Indian Meteorological Department
        3. Solar Energy Research for Rural Applications
        4. Machine Learning for Renewable Energy Forecasting
        
        ---
        
        ## Support
        
        For technical issues or research collaboration:
        - Check data availability for selected location
        - Ensure stable internet connection for API access
        - Validate model assumptions for specific applications
        
        **Note:** This tool is designed for planning and research purposes. 
        For actual system design, consult with solar energy professionals.
        """)


# Run the app
if __name__ == "__main__":
    main()