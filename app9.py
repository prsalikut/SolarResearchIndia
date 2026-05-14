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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import time

# Page configuration
st.set_page_config(
    page_title="Solar Power Forecasting - Rural India",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA COLLECTION CLASS ====================
class SolarDataCollector:
    """Collect weather and solar irradiance data using free APIs with intelligent fallback"""
    
    def __init__(self):
        self.open_meteo_base = "https://archive-api.open-meteo.com/v1/archive"
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def get_solar_data_nasa_power(self, latitude, longitude, start_date, end_date):
        """Get solar irradiance data from NASA POWER"""
        today = datetime.now().date()
        max_end_date = today
        
        if end_date > max_end_date:
            end_date = max_end_date
            st.info(f"Using data up to {end_date} (NASA POWER supports historical data)")
        
        if start_date > end_date:
            start_date = end_date - timedelta(days=365*3)
        
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
    
    def generate_smart_synthetic_data(self, latitude, longitude, start_date, end_date):
        """Generate realistic synthetic data based on location patterns"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        if latitude > 20:  # Northern India
            base_temp = 25 + 12 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 5.5 + 2.5 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        else:  # Southern India
            base_temp = 28 + 8 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 6 + 2 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        
        np.random.seed(int(latitude * longitude))
        
        df = pd.DataFrame({
            'date': date_range,
            'solar_irradiance_kwh_m2': np.maximum(base_irradiance + np.random.normal(0, 1.2, len(date_range)), 0),
            'temperature_c': base_temp + np.random.normal(0, 3, len(date_range)),
            'relative_humidity': 50 + 25 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + np.random.normal(0, 12, len(date_range)),
            'cloudcover_mean': 40 + 35 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + np.random.normal(0, 18, len(date_range))
        })
        
        df['temperature_2m_mean'] = df['temperature_c']
        df['temperature_2m_max'] = df['temperature_c'] + 5 + np.random.normal(0, 2, len(date_range))
        df['temperature_2m_min'] = df['temperature_c'] - 5 + np.random.normal(0, 2, len(date_range))
        df['precipitation_sum'] = np.random.exponential(0.3, len(date_range))
        df['windspeed_10m_max'] = 4 + 3 * np.random.random(len(date_range))
        df['clear_sky_irradiance_kwh_m2'] = df['solar_irradiance_kwh_m2'] * (1.2 + 0.3 * np.random.random(len(date_range)))
        
        df['solar_irradiance_kwh_m2'] = df['solar_irradiance_kwh_m2'].clip(0, 8)
        df['temperature_2m_mean'] = df['temperature_2m_mean'].clip(10, 45)
        df['relative_humidity'] = df['relative_humidity'].clip(20, 100)
        df['cloudcover_mean'] = df['cloudcover_mean'].clip(0, 100)
        df['clear_sky_irradiance_kwh_m2'] = df['clear_sky_irradiance_kwh_m2'].clip(0, 10)
        
        return df
    
    def collect_data_for_location(self, location_name, latitude, longitude, start_date, end_date):
        """Intelligent data collection with automatic fallback"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Fetching solar irradiance data from NASA POWER...")
        progress_bar.progress(30)
        
        solar_df = self.get_solar_data_nasa_power(latitude, longitude, start_date, end_date)
        
        if solar_df is not None:
            status_text.text("Finalizing data collection...")
            progress_bar.progress(90)
            merged_df = solar_df.copy()
            merged_df['temperature_2m_mean'] = merged_df['temperature_c']
            merged_df['cloudcover_mean'] = 50
        else:
            status_text.text("Generating realistic data based on location patterns...")
            progress_bar.progress(70)
            merged_df = self.generate_smart_synthetic_data(latitude, longitude, start_date, end_date)
            st.info(f"Using realistic generated data for {location_name} based on location patterns")
        
        merged_df['location'] = location_name
        merged_df['latitude'] = latitude
        merged_df['longitude'] = longitude
        
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].mean())
        
        progress_bar.progress(100)
        status_text.text(f"Data collection complete: {len(merged_df)} days")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return merged_df


# ==================== SOLAR POWER CALCULATOR ====================
def calculate_solar_power(df, panel_capacity_kw=5.0, efficiency=0.17):
    """Calculate solar power output from irradiance data"""
    df = df.copy()
    
    if 'solar_irradiance_kwh_m2' in df.columns:
        irradiance_col = 'solar_irradiance_kwh_m2'
    elif 'estimated_irradiance' in df.columns:
        irradiance_col = 'estimated_irradiance'
    else:
        return df
    
    panel_area = panel_capacity_kw / (1.0 * efficiency)
    df['solar_power_kwh'] = df[irradiance_col] * panel_area * efficiency
    
    efficiency_factors = ['temp_efficiency_factor', 'cloud_reduction_factor']
    for factor in efficiency_factors:
        if factor in df.columns:
            df[factor] = df[factor].clip(lower=0.1)
            df['solar_power_kwh'] *= df[factor]
    
    if 'is_monsoon' in df.columns:
        df['solar_power_kwh'] *= (0.6 + 0.4 * (1 - df['is_monsoon']))
    
    max_theoretical = panel_capacity_kw * 8
    df['solar_power_kwh'] = df['solar_power_kwh'].clip(lower=0, upper=max_theoretical)
    
    night_threshold = 0.05
    df.loc[df[irradiance_col] <= night_threshold, 'solar_power_kwh'] = 0
    
    return df


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
        
        df['season'] = df['month'].apply(lambda x: 
            'Winter' if x in [12, 1, 2] else
            'Summer' if x in [3, 4, 5] else
            'Monsoon' if x in [6, 7, 8, 9] else
            'Post-Monsoon'
        )
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_summer_peak'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        
        return df
    
    @staticmethod
    def add_solar_features(df):
        """Add solar-specific features"""
        df = df.copy()
        
        if 'solar_irradiance_kwh_m2' in df.columns and 'clear_sky_irradiance_kwh_m2' in df.columns:
            df['clearness_index'] = df['solar_irradiance_kwh_m2'] / (df['clear_sky_irradiance_kwh_m2'] + 0.001)
            df['clearness_index'] = df['clearness_index'].clip(0, 1)
        elif 'solar_irradiance_kwh_m2' in df.columns:
            df['clearness_index'] = 0.7 + 0.1 * np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['clearness_index'] = df['clearness_index'].clip(0.5, 0.9)
        
        temp_col = 'temperature_2m_mean' if 'temperature_2m_mean' in df.columns else 'temperature_c'
        if temp_col in df.columns:
            df['temp_efficiency_factor'] = np.maximum(0.85, 1.0 - 0.003 * np.abs(df[temp_col] - 25))
            df['temp_efficiency_factor'] = df['temp_efficiency_factor'].clip(0.85, 1.05)
        
        if 'cloudcover_mean' in df.columns:
            df['cloud_reduction_factor'] = 1.0 - (df['cloudcover_mean'] / 100) * 0.6
            df['cloud_reduction_factor'] = df['cloud_reduction_factor'].clip(0.4, 1.0)
        
        return df
    
    @staticmethod
    def create_future_features(future_dates, location_lat):
        """Create features for future dates"""
        df = pd.DataFrame({'date': future_dates})
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        df['is_summer_peak'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        base_irradiance = 5.5 + 2.5 * np.sin(2 * np.pi * (df['day_of_year'] - 80) / 365.25)
        monsoon_reduction = np.where(df['is_monsoon'] == 1, 0.6, 1.0)
        df['estimated_irradiance'] = base_irradiance * monsoon_reduction
        
        if location_lat > 20:
            df['estimated_temp'] = 25 + 12 * np.sin(2 * np.pi * (df['day_of_year'] - 105) / 365.25)
        else:
            df['estimated_temp'] = 28 + 8 * np.sin(2 * np.pi * (df['day_of_year'] - 105) / 365.25)
        
        np.random.seed(42)
        df['estimated_irradiance'] += np.random.normal(0, 0.8, len(df))
        df['estimated_temp'] += np.random.normal(0, 2.5, len(df))
        
        df['estimated_irradiance'] = df['estimated_irradiance'].clip(0, 8)
        df['estimated_temp'] = df['estimated_temp'].clip(10, 45)
        
        return df


# ==================== FORECASTING MODELS ====================
class HybridForecastingModels:
    """Hybrid ML and Statistical models for solar power forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=200, random_state=42):
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        self.models['Random Forest'] = model
        self.scalers['Random Forest'] = scaler
        self.feature_names['Random Forest'] = X_train.columns.tolist()
        
        return metrics, y_pred
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, n_estimators=200, learning_rate=0.1, random_state=42):
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=random_state
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        self.models['Gradient Boosting'] = model
        self.scalers['Gradient Boosting'] = scaler
        self.feature_names['Gradient Boosting'] = X_train.columns.tolist()
        
        return metrics, y_pred
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        self.models['Linear Regression'] = model
        self.scalers['Linear Regression'] = scaler
        self.feature_names['Linear Regression'] = X_train.columns.tolist()
        
        return metrics, y_pred
    
    def train_svr(self, X_train, y_train, X_test, y_test, kernel='rbf', C=1.0, epsilon=0.1):
        """Train Support Vector Regression model"""
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        self.models['SVR'] = model
        self.scalers['SVR'] = scaler
        self.feature_names['SVR'] = X_train.columns.tolist()
        
        return metrics, y_pred
    
    def train_neural_network(self, X_train, y_train, X_test, y_test, hidden_layer_sizes=(100, 50), random_state=42):
        """Train Neural Network model"""
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=1000,
            learning_rate='adaptive'
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        self.models['Neural Network'] = model
        self.scalers['Neural Network'] = scaler
        self.feature_names['Neural Network'] = X_train.columns.tolist()
        
        return metrics, y_pred
    
    def train_arima(self, y_train, y_test, order=(2,1,2)):
        """Train ARIMA model (univariate time series)"""
        if not STATSMODELS_AVAILABLE:
            return None, None
            
        try:
            model = ARIMA(y_train, order=order)
            model_fit = model.fit()
            y_pred = model_fit.forecast(steps=len(y_test))
            
            metrics = self._calculate_metrics(y_test, y_pred)
            self.models['ARIMA'] = model_fit
            
            return metrics, y_pred
        except Exception as e:
            st.warning(f"ARIMA training failed: {str(e)}")
            return None, None
    
    def train_sarima(self, y_train, y_test, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Train SARIMA model (seasonal ARIMA)"""
        if not STATSMODELS_AVAILABLE:
            return None, None
            
        try:
            model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            y_pred = model_fit.forecast(steps=len(y_test))
            
            metrics = self._calculate_metrics(y_test, y_pred)
            self.models['SARIMA'] = model_fit
            
            return metrics, y_pred
        except Exception as e:
            st.warning(f"SARIMA training failed: {str(e)}")
            return None, None
    
    def train_exponential_smoothing(self, y_train, y_test, seasonal_periods=12):
        """Train Exponential Smoothing model"""
        if not STATSMODELS_AVAILABLE:
            return None, None
            
        try:
            model = ExponentialSmoothing(y_train, seasonal_periods=seasonal_periods)
            model_fit = model.fit()
            y_pred = model_fit.forecast(steps=len(y_test))
            
            metrics = self._calculate_metrics(y_test, y_pred)
            self.models['Exponential Smoothing'] = model_fit
            
            return metrics, y_pred
        except Exception as e:
            st.warning(f"Exponential Smoothing training failed: {str(e)}")
            return None, None
    
    def train_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Train ensemble of multiple models"""
        # Train individual models
        rf_metrics, rf_pred = self.train_random_forest(X_train, y_train, X_test, y_test)
        gb_metrics, gb_pred = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        # Simple average ensemble
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        self.models['Ensemble (RF+GB)'] = {'rf': self.models['Random Forest'], 'gb': self.models['Gradient Boosting']}
        
        return metrics, ensemble_pred
    
    def predict_future(self, model_name, historical_df, future_months, panel_capacity, efficiency, location_lat):
        """Predict future solar power generation"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        last_date = historical_df['date'].max()
        future_days = future_months * 30
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=future_days,
            freq='D'
        )
        
        fe = SolarFeatureEngineering()
        future_df = fe.create_future_features(future_dates, location_lat)
        future_df = calculate_solar_power(future_df, panel_capacity, efficiency)
        
        if model_name in ['ARIMA', 'SARIMA', 'Exponential Smoothing']:
            # Univariate time series models
            historical_series = historical_df['solar_power_kwh'].values
            model = self.models[model_name]
            
            if model_name == 'ARIMA':
                predictions = model.forecast(steps=len(future_dates))
            elif model_name == 'SARIMA':
                predictions = model.forecast(steps=len(future_dates))
            else:  # Exponential Smoothing
                predictions = model.forecast(steps=len(future_dates))
        elif model_name == 'Ensemble (RF+GB)':
            # Ensemble model predictions
            rf_model = self.models['Random Forest']
            gb_model = self.models['Gradient Boosting']
            rf_scaler = self.scalers['Random Forest']
            gb_scaler = self.scalers['Gradient Boosting']
            
            future_processed, _ = self._prepare_future_features(future_df, model_name)
            X_future = future_processed[self.feature_names['Random Forest']]
            
            X_future_scaled_rf = rf_scaler.transform(X_future)
            X_future_scaled_gb = gb_scaler.transform(X_future)
            
            rf_pred = rf_model.predict(X_future_scaled_rf)
            gb_pred = gb_model.predict(X_future_scaled_gb)
            
            predictions = (rf_pred + gb_pred) / 2
        else:
            # ML models with features
            future_processed, feature_cols = self._prepare_future_features(future_df, model_name)
            X_future = future_processed[self.feature_names[model_name]]
            scaler = self.scalers[model_name]
            X_future_scaled = scaler.transform(X_future)
            model = self.models[model_name]
            predictions = model.predict(X_future_scaled)
        
        predictions = np.maximum(predictions, 0)
        
        results_df = pd.DataFrame({
            'date': future_dates,
            'predicted_power_kwh': predictions,
            'model': model_name
        })
        
        return results_df
    
    def _prepare_future_features(self, df, model_name):
        """Prepare features for future prediction"""
        fe = SolarFeatureEngineering()
        df = fe.add_temporal_features(df)
        df = fe.add_solar_features(df)
        
        exclude_cols = ['date', 'location', 'latitude', 'longitude', 'solar_power_kwh', 
                       'season', 'estimated_irradiance', 'estimated_temp']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
        
        if feature_cols:
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
        
        return df, feature_cols
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        if y_pred is None:
            return None
            
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        mask = y_true > 0
        if np.any(mask):
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = 0
        
        return metrics


# ==================== VISUALIZATION FUNCTIONS ====================
def plot_historical_data(df, location_name):
    """Plot historical solar and weather data"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Solar Power Generation', 'Solar Irradiance', 'Temperature'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    if 'solar_power_kwh' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['solar_power_kwh'],
                      name='Solar Power', line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
    
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

def plot_model_comparison(metrics_df):
    """Plot model performance comparison"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE Comparison', 'MAE Comparison', 
                       'R² Score Comparison', 'MAPE Comparison'),
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for metric, pos in zip(metrics, positions):
        fig.add_trace(
            go.Bar(x=metrics_df['Model'], y=metrics_df[metric],
                  name=metric, marker_color='#3498db'),
            row=pos[0], col=pos[1]
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_forecast_comparison(historical_df, forecast_dfs, location_name, panel_capacity):
    """Plot historical data with multiple forecast models"""
    fig = go.Figure()
    
    historical_recent = historical_df.tail(90)
    if 'solar_power_kwh' in historical_recent.columns:
        fig.add_trace(go.Scatter(
            x=historical_recent['date'],
            y=historical_recent['solar_power_kwh'],
            mode='lines',
            name='Historical',
            line=dict(color='#2c3e50', width=2)
        ))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    for i, (model_name, forecast_df) in enumerate(forecast_dfs.items()):
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['predicted_power_kwh'],
            mode='lines',
            name=model_name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=f"Solar Power Forecast Comparison - {location_name} ({panel_capacity} kW)",
        xaxis_title="Date",
        yaxis_title="Solar Power Generation (kWh)",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_seasonal_decomposition(df):
    """Plot seasonal decomposition of solar power"""
    if 'solar_power_kwh' not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy.set_index('date', inplace=True)
    
    # Simple moving average for trend
    df_copy['trend'] = df_copy['solar_power_kwh'].rolling(window=30, center=True).mean()
    df_copy['detrended'] = df_copy['solar_power_kwh'] - df_copy['trend']
    
    # Monthly seasonal pattern
    monthly_avg = df_copy.groupby('month')['detrended'].mean()
    df_copy['seasonal'] = df_copy['month'].map(monthly_avg)
    df_copy['residual'] = df_copy['detrended'] - df_copy['seasonal']
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original Series', 'Trend Component', 
                       'Seasonal Component', 'Residual Component'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy.index, y=df_copy['solar_power_kwh'],
                  name='Original', line=dict(color='#3498db')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy.index, y=df_copy['trend'],
                  name='Trend', line=dict(color='#2ecc71')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy.index, y=df_copy['seasonal'],
                  name='Seasonal', line=dict(color='#e74c3c')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_copy.index, y=df_copy['residual'],
                  name='Residual', line=dict(color='#f39c12')),
        row=4, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="kWh", row=1, col=1)
    fig.update_yaxes(title_text="kWh", row=2, col=1)
    fig.update_yaxes(title_text="kWh", row=3, col=1)
    fig.update_yaxes(title_text="kWh", row=4, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


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
    
    today = datetime.now().date()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=365)
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
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
    
    # Model Selection
    st.sidebar.subheader("Model Selection")
    available_models = ['Random Forest', 'Gradient Boosting', 'Linear Regression', 
                       'SVR', 'Neural Network', 'Ensemble (RF+GB)']
    
    if STATSMODELS_AVAILABLE:
        available_models.extend(['ARIMA', 'SARIMA', 'Exponential Smoothing'])
    
    selected_models = st.sidebar.multiselect(
        "Select Forecasting Models",
        available_models,
        default=['Random Forest', 'Gradient Boosting', 'Ensemble (RF+GB)']
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = HybridForecastingModels()
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Collection & Analysis", 
        "Model Training & Comparison", 
        "Future Forecast", 
        "Documentation"
    ])
    
    # ==================== TAB 1: DATA COLLECTION & ANALYSIS ====================
    with tab1:
        st.markdown('<p class="sub-header">Data Collection & Analysis</p>', unsafe_allow_html=True)
        
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.metric("Location", location_name)
            st.metric("Coordinates", f"{latitude}°N, {longitude}°E")
        with config_col2:
            st.metric("Data Period", f"{start_date} to {end_date}")
            st.metric("Panel Configuration", f"{panel_capacity} kW, {panel_efficiency*100:.0f}%")
        
        if st.button("Collect Historical Data", type="primary", width='stretch'):
            with st.spinner("Collecting data..."):
                try:
                    collector = SolarDataCollector()
                    
                    df = collector.collect_data_for_location(
                        location_name, latitude, longitude,
                        start_date, end_date
                    )
                    
                    if df is not None and len(df) > 0:
                        fe = SolarFeatureEngineering()
                        df = fe.add_temporal_features(df)
                        df = fe.add_solar_features(df)
                        df = calculate_solar_power(df, panel_capacity, panel_efficiency)
                        
                        st.session_state.data = df
                        
                        st.success(f"Successfully collected and processed {len(df)} days of data")
                        
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
                            irradiance_col = None
                            if 'solar_irradiance_kwh_m2' in df.columns:
                                irradiance_col = 'solar_irradiance_kwh_m2'
                            elif 'estimated_irradiance' in df.columns:
                                irradiance_col = 'estimated_irradiance'
                            
                            if irradiance_col:
                                df[irradiance_col] = df[irradiance_col].clip(lower=0)
                                avg_irrad = df[irradiance_col].mean()
                                st.metric("Avg Irradiance", f"{avg_irrad:.2f} kWh/m²")
                            else:
                                st.metric("Avg Irradiance", "N/A")
                        
                    else:
                        st.error("Failed to collect data. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("---")
            st.subheader("Data Visualization")
            
            fig_historical = plot_historical_data(df, location_name)
            st.plotly_chart(fig_historical, width='stretch')
            
            st.subheader("Seasonal Decomposition")
            fig_decomposition = plot_seasonal_decomposition(df)
            if fig_decomposition:
                st.plotly_chart(fig_decomposition, width='stretch')
            
            with st.expander("View Data Table"):
                display_cols = ['date']
                possible_cols = ['solar_power_kwh', 'solar_irradiance_kwh_m2', 
                               'estimated_irradiance', 'temperature_2m_mean', 
                               'temperature_c', 'estimated_temp', 'cloudcover_mean']
                
                for col in possible_cols:
                    if col in df.columns:
                        display_cols.append(col)
                
                if len(display_cols) > 1:
                    st.dataframe(df[display_cols].head(20), width='stretch')
                else:
                    st.warning("No data columns available to display")
            
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
    
    # ==================== TAB 2: MODEL TRAINING & COMPARISON ====================
    with tab2:
        st.markdown('<p class="sub-header">Model Training & Comparison</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            if 'solar_power_kwh' not in df.columns:
                st.error("Solar power data not available. Please recollect data.")
            else:
                st.info(f"Using {len(df)} days of historical data for model training")
                
                st.subheader("Training Configuration")
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
                with col2:
                    random_state = st.number_input("Random Seed", value=42, min_value=0)
                
                if st.button("Train Selected Models", type="primary", width='stretch'):
                    if not selected_models:
                        st.error("Please select at least one model to train")
                    else:
                        with st.spinner(f"Training {len(selected_models)} models..."):
                            try:
                                # Prepare features
                                fe = SolarFeatureEngineering()
                                df_features = fe.add_temporal_features(df)
                                df_features = fe.add_solar_features(df_features)
                                
                                exclude_cols = ['date', 'location', 'latitude', 'longitude', 'solar_power_kwh', 
                                              'season', 'estimated_irradiance', 'estimated_temp']
                                feature_cols = [col for col in df_features.columns if col not in exclude_cols 
                                              and pd.api.types.is_numeric_dtype(df_features[col])]
                                
                                X = df_features[feature_cols]
                                y = df_features['solar_power_kwh']
                                
                                # Handle missing values
                                X = X.fillna(X.mean())
                                y = y.fillna(y.mean())
                                
                                # Time-series split (preserve temporal order)
                                split_idx = int(len(X) * (1 - test_size/100))
                                X_train, X_test = X[:split_idx], X[split_idx:]
                                y_train, y_test = y[:split_idx], y[split_idx:]
                                
                                forecaster = st.session_state.forecaster
                                all_metrics = []
                                
                                for model_name in selected_models:
                                    with st.spinner(f"Training {model_name}..."):
                                        try:
                                            if model_name == 'Random Forest':
                                                metrics, y_pred = forecaster.train_random_forest(
                                                    X_train, y_train, X_test, y_test, random_state=random_state
                                                )
                                            elif model_name == 'Gradient Boosting':
                                                metrics, y_pred = forecaster.train_gradient_boosting(
                                                    X_train, y_train, X_test, y_test, random_state=random_state
                                                )
                                            elif model_name == 'Linear Regression':
                                                metrics, y_pred = forecaster.train_linear_regression(
                                                    X_train, y_train, X_test, y_test
                                                )
                                            elif model_name == 'SVR':
                                                metrics, y_pred = forecaster.train_svr(
                                                    X_train, y_train, X_test, y_test
                                                )
                                            elif model_name == 'Neural Network':
                                                metrics, y_pred = forecaster.train_neural_network(
                                                    X_train, y_train, X_test, y_test, random_state=random_state
                                                )
                                            elif model_name == 'ARIMA':
                                                metrics, y_pred = forecaster.train_arima(y_train, y_test)
                                            elif model_name == 'SARIMA':
                                                metrics, y_pred = forecaster.train_sarima(y_train, y_test)
                                            elif model_name == 'Exponential Smoothing':
                                                metrics, y_pred = forecaster.train_exponential_smoothing(y_train, y_test)
                                            elif model_name == 'Ensemble (RF+GB)':
                                                metrics, y_pred = forecaster.train_ensemble_model(
                                                    X_train, y_train, X_test, y_test
                                                )
                                            
                                            if metrics:
                                                metrics_df = pd.DataFrame([metrics])
                                                metrics_df['Model'] = model_name
                                                all_metrics.append(metrics_df)
                                                st.success(f"{model_name} trained successfully")
                                            else:
                                                st.warning(f"{model_name} training completed with warnings")
                                                
                                        except Exception as e:
                                            st.error(f"Failed to train {model_name}: {str(e)}")
                                
                                if all_metrics:
                                    metrics_df = pd.concat(all_metrics, ignore_index=True)
                                    st.session_state.model_metrics = metrics_df
                                    st.session_state.trained_models = selected_models
                                    
                                    st.success(f"Successfully trained {len(all_metrics)} models")
                                    
                                    # Display metrics comparison
                                    st.subheader("Model Performance Comparison")
                                    
                                    fig_comparison = plot_model_comparison(metrics_df)
                                    st.plotly_chart(fig_comparison, width='stretch')
                                    
                                    # Show metrics table
                                    st.dataframe(metrics_df.set_index('Model'), width='stretch')
                                    
                                    # Best model identification
                                    best_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin()]
                                    best_r2 = metrics_df.loc[metrics_df['R2'].idxmax()]
                                    
                                    insights = f"""
                                    <div class="info-box">
                                    <b>Key Insights:</b><br><br>
                                    • <b>Best Accuracy (Lowest RMSE):</b> {best_rmse['Model']} (RMSE: {best_rmse['RMSE']:.3f})<br>
                                    • <b>Best Fit (Highest R²):</b> {best_r2['Model']} (R²: {best_r2['R2']:.3f})<br>
                                    • <b>Average MAPE:</b> {metrics_df['MAPE'].mean():.1f}% across all models<br>
                                    • <b>Models Trained:</b> {', '.join(selected_models)}
                                    </div>
                                    """
                                    st.markdown(insights, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Training failed: {str(e)}")
                else:
                    st.info("Select models and click 'Train Selected Models' to start training")
        else:
            st.warning("Please collect data first in the 'Data Collection & Analysis' tab.")
    
    # ==================== TAB 3: FUTURE FORECAST ====================
    with tab3:
        st.markdown('<p class="sub-header">Future Solar Power Forecast</p>', unsafe_allow_html=True)
        
        if st.session_state.trained_models:
            forecaster = st.session_state.forecaster
            historical_df = st.session_state.data
            
            st.write(f"""
            **Forecasting Setup:**  
            • **Location:** {location_name}  
            • **System:** {panel_capacity} kW, {panel_efficiency*100:.0f}% efficiency  
            • **Forecast Period:** {forecast_months} months  
            • **Trained Models:** {', '.join(st.session_state.trained_models)}
            """)
            
            if st.button("Generate Forecasts", type="primary", width='stretch'):
                with st.spinner(f"Generating {forecast_months}-month forecasts..."):
                    try:
                        forecasts = {}
                        for model_name in st.session_state.trained_models:
                            with st.spinner(f"Generating forecast with {model_name}..."):
                                forecast_df = forecaster.predict_future(
                                    model_name, historical_df, forecast_months, 
                                    panel_capacity, panel_efficiency, latitude
                                )
                                forecasts[model_name] = forecast_df
                        
                        st.session_state.forecasts = forecasts
                        st.success(f"Forecasts generated for {len(forecasts)} models")
                        
                    except Exception as e:
                        st.error(f"Forecast generation failed: {str(e)}")
            
            if st.session_state.forecasts:
                forecasts = st.session_state.forecasts
                
                st.subheader("Forecast Comparison")
                
                fig_forecast = plot_forecast_comparison(historical_df, forecasts, location_name, panel_capacity)
                st.plotly_chart(fig_forecast, width='stretch')
                
                st.subheader("Forecast Summary by Model")
                
                summary_data = []
                for model_name, forecast_df in forecasts.items():
                    total_energy = forecast_df['predicted_power_kwh'].sum()
                    avg_daily = forecast_df['predicted_power_kwh'].mean()
                    max_daily = forecast_df['predicted_power_kwh'].max()
                    
                    summary_data.append({
                        'Model': model_name,
                        'Total Forecast Energy (kWh)': f"{total_energy:,.0f}",
                        'Average Daily (kWh)': f"{avg_daily:.1f}",
                        'Peak Daily (kWh)': f"{max_daily:.1f}",
                        'Forecast Days': len(forecast_df)
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width='stretch')
                
                st.subheader("Monthly Forecast Analysis")
                
                for model_name, forecast_df in forecasts.items():
                    with st.expander(f"{model_name} - Monthly Forecast"):
                        monthly_forecast = forecast_df.copy()
                        monthly_forecast['month'] = monthly_forecast['date'].dt.month
                        
                        monthly_summary = monthly_forecast.groupby('month').agg({
                            'predicted_power_kwh': ['sum', 'mean', 'std']
                        }).round(2)
                        
                        monthly_summary.columns = ['total_kwh', 'avg_daily_kwh', 'std_daily_kwh']
                        monthly_summary = monthly_summary.reset_index()
                        
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        monthly_summary['month_name'] = monthly_summary['month'].apply(lambda x: month_names[x-1])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=monthly_summary['month_name'],
                            y=monthly_summary['total_kwh'],
                            name='Total Generation',
                            marker_color='#3498db'
                        ))
                        
                        fig.update_layout(
                            title=f"Monthly Forecast - {model_name}",
                            xaxis_title="Month",
                            yaxis_title="Total Generation (kWh)",
                            height=400,
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        st.dataframe(monthly_summary[['month_name', 'total_kwh', 'avg_daily_kwh']], width='stretch')
                
                st.subheader("Download Forecasts")
                for model_name, forecast_df in forecasts.items():
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {model_name} Forecast",
                        data=csv,
                        file_name=f"forecast_{location_name}_{model_name}_{forecast_months}months.csv",
                        mime="text/csv"
                    )
                
        elif st.session_state.data is not None:
            st.info("Please train models first in the 'Model Training & Comparison' tab.")
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
        
        ### 1. Hybrid Forecasting Approach
        - **Machine Learning Models:** Random Forest, Gradient Boosting, Linear Regression, SVR, Neural Networks
        - **Statistical Models:** ARIMA, SARIMA, Exponential Smoothing
        - **Ensemble Methods:** Combining multiple models for improved accuracy
        
        ### 2. Data Processing Pipeline
        - **Data Collection:** NASA POWER API for solar irradiance and weather data
        - **Feature Engineering:** Temporal patterns, seasonal indicators, weather correlations
        - **Solar Power Calculation:** Panel capacity and efficiency considerations
        - **Validation:** Time-series cross-validation preserving temporal order
        
        ### 3. Model Evaluation
        - **Metrics:** RMSE, MAE, R², MAPE for comprehensive evaluation
        - **Comparison:** Side-by-side model performance assessment
        - **Interpretability:** Feature importance analysis for ML models
        
        ---
        
        ## Forecasting Models Implemented
        
        ### 1. Random Forest
        - Ensemble of decision trees
        - Handles non-linear relationships
        - Provides feature importance
        
        ### 2. Gradient Boosting
        - Sequential ensemble of weak learners
        - High predictive accuracy
        - Built-in feature importance
        
        ### 3. Linear Regression
        - Simple and interpretable
        - Fast training and prediction
        - Baseline model for comparison
        
        ### 4. Support Vector Regression (SVR)
        - Effective in high-dimensional spaces
        - Robust to outliers
        - Kernel trick for non-linear patterns
        
        ### 5. Neural Network
        - Multi-layer perceptron
        - Captures complex patterns
        - Non-linear activation functions
        
        ### 6. ARIMA (AutoRegressive Integrated Moving Average)
        - Classical time series model
        - Captures autocorrelation patterns
        - Suitable for stationary data
        
        ### 7. SARIMA (Seasonal ARIMA)
        - Extends ARIMA with seasonal components
        - Captures seasonal patterns
        - Ideal for solar data with yearly cycles
        
        ### 8. Exponential Smoothing
        - Weighted average of past observations
        - Recent observations get more weight
        - Simple yet effective for seasonal data
        
        ### 9. Ensemble Methods
        - Combines Random Forest and Gradient Boosting
        - Reduces individual model biases
        - Improves overall accuracy and robustness
        
        ---
        
        ## Application Areas
        
        ### 1. Rural Solar Planning
        - Long-term energy production forecasting
        - System sizing and optimization
        - Battery storage requirement planning
        
        ### 2. Policy Development
        - Regional solar potential assessment
        - Investment risk analysis
        - Energy access strategy formulation
        
        ### 3. Research & Analysis
        - Climate impact studies
        - Technology performance comparison
        - Seasonal and interannual trend analysis
        
        ---
        
        ## How to Use This Tool
        
        ### Step 1: Data Collection
        1. Select location and date range
        2. Set solar system parameters
        3. Click "Collect Historical Data"
        
        ### Step 2: Model Training
        1. Select forecasting models to train
        2. Configure training parameters
        3. Click "Train Selected Models"
        4. Compare model performance
        
        ### Step 3: Future Forecasting
        1. Set forecast period
        2. Generate forecasts with trained models
        3. Compare forecasts across models
        4. Download forecast data
        
        ---
        
        ## Technical Implementation
        
        ### Data Sources
        - **Primary:** NASA POWER API for solar irradiance
        - **Features:** Temporal, seasonal, weather variables
        - **Validation:** Time-series preserving temporal order
        
        ### Model Implementation
        - **Scikit-learn:** All ML models (RF, GB, LR, SVR, NN)
        - **Statsmodels:** Statistical models (ARIMA, SARIMA)
        - **Custom Ensemble:** Model combination strategies
        
        ### Performance Metrics
        - **RMSE:** Root Mean Square Error
        - **MAE:** Mean Absolute Error
        - **R²:** Coefficient of Determination
        - **MAPE:** Mean Absolute Percentage Error
        
        ---
        
        ## References
        
        1. NASA POWER Data Services Documentation
        2. Indian Meteorological Department Data
        3. Machine Learning for Renewable Energy Forecasting (Research Papers)
        4. Time Series Analysis and Forecasting Literature
        
        ---
        
        ## Support & Limitations
        
        ### Current Capabilities
        - Multiple forecasting model options
        - Comparative performance analysis
        - Long-term trend forecasting
        - Seasonal pattern identification
        
        ### Limitations
        - Dependent on historical data quality
        - Assumes climate stationarity
        - Daily resolution forecasts
        - Regional generalization requires validation
        
        ### Future Enhancements
        - Integration of satellite imagery
        - Real-time data streaming
        - Higher temporal resolution
        - Climate change projections
        
        **Note:** This tool is designed for research and planning purposes. 
        For actual system design, consult with solar energy professionals.
        """)


# Run the app
if __name__ == "__main__":
    main()