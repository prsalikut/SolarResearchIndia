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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR

# Optional imports
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import time
import json
import pickle
import os
from pathlib import Path

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Solar Power Forecasting - Enhanced",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 10px 0;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: #999;
        margin-top: 5px;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 10px 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 30px 0;
    }
    
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .model-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .rf-badge { background: #e3f2fd; color: #1565c0; }
    .xgb-badge { background: #fff3e0; color: #ef6c00; }
    .gb-badge { background: #e0f7fa; color: #006064; }
    .svr-badge { background: #fce4ec; color: #c2185b; }
    .lstm-badge { background: #f3e5f5; color: #7b1fa2; }
    .ensemble-badge { background: #e8f5e9; color: #2e7d32; }
    
    .progress-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA COLLECTOR CLASS ====================
class EnhancedSolarDataCollector:
    """Enhanced data collector with caching and error handling"""
    
    def __init__(self, api_cache_dir="api_cache"):
        self.open_meteo_base = "https://archive-api.open-meteo.com/v1/archive"
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.api_cache_dir = Path(api_cache_dir)
        self.api_cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, source, latitude, longitude, start_date, end_date):
        """Generate cache key for API requests"""
        return f"{source}_{latitude}_{longitude}_{start_date}_{end_date}.pkl"
    
    def _load_from_cache(self, cache_key):
        """Load data from cache if available"""
        cache_file = self.api_cache_dir / cache_key
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    return data
            except Exception as e:
                st.warning(f"Cache read error: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key, data):
        """Save data to cache"""
        try:
            cache_file = self.api_cache_dir / cache_key
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.warning(f"Cache write error: {e}")
    
    def get_enhanced_weather_data(self, latitude, longitude, start_date, end_date):
        """Fetch weather data from Open-Meteo API"""
        cache_key = self._get_cache_key("openmeteo", latitude, longitude, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            st.info("✓ Using cached weather data")
            return cached_data
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ('temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                     'apparent_temperature_max,apparent_temperature_min,'
                     'precipitation_sum,rain_sum,snowfall_sum,'
                     'windspeed_10m_max,windspeed_10m_mean,windgusts_10m_max,'
                     'shortwave_radiation_sum,diffuse_radiation_sum,'
                     'et0_fao_evapotranspiration,relative_humidity_2m_mean,'
                     'cloudcover_mean,pressure_msl_mean,'
                     'sunshine_duration,weathercode'),
            'timezone': 'Asia/Kolkata',
            'models': 'best_match'
        }
        
        try:
            with st.spinner("Fetching weather data from Open-Meteo..."):
                response = requests.get(self.open_meteo_base, params=params, timeout=45)
                response.raise_for_status()
                data = response.json()
                
                if 'daily' in data:
                    df = pd.DataFrame(data['daily'])
                    df['date'] = pd.to_datetime(df['time'])
                    df = df.drop('time', axis=1)
                    
                    self._save_to_cache(cache_key, df)
                    st.success(f"✓ Weather data collected: {len(df)} days")
                    return df
        except requests.exceptions.Timeout:
            st.error("⚠ Request timeout. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠ Network error: {e}")
        except Exception as e:
            st.error(f"⚠ Data processing error: {e}")
        
        return None
    
    def get_enhanced_solar_data(self, latitude, longitude, start_date, end_date):
        """Fetch solar data from NASA POWER API"""
        cache_key = self._get_cache_key("nasapower", latitude, longitude, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None:
            st.info("✓ Using cached solar data")
            return cached_data
        
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        
        params = {
            'parameters': ('ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,ALLSKY_KT,'
                          'T2M_MAX,T2M_MIN,T2M_DEW,TS,WS10M_MAX,WS10M,'
                          'PS,CLOUD_AMT,RH2M,PRECTOTCORR,ALLSKY_SFC_PAR_TOT'),
            'community': 'RE',
            'longitude': longitude,
            'latitude': latitude,
            'start': start,
            'end': end,
            'format': 'JSON'
        }
        
        try:
            with st.spinner("Fetching solar data from NASA POWER..."):
                response = requests.get(self.nasa_power_base, params=params, timeout=90)
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
                    
                    column_mapping = {
                        'ALLSKY_SFC_SW_DWN': 'solar_irradiance_kwh_m2',
                        'CLRSKY_SFC_SW_DWN': 'clear_sky_irradiance_kwh_m2',
                        'ALLSKY_KT': 'clearness_index',
                        'T2M_MAX': 'temperature_max_c',
                        'T2M_MIN': 'temperature_min_c',
                        'T2M_DEW': 'dew_point_c',
                        'TS': 'surface_temp_c',
                        'WS10M_MAX': 'wind_speed_max_ms',
                        'WS10M': 'wind_speed_ms',
                        'PS': 'surface_pressure',
                        'CLOUD_AMT': 'cloud_amount',
                        'RH2M': 'relative_humidity',
                        'PRECTOTCORR': 'precipitation_mm',
                        'ALLSKY_SFC_PAR_TOT': 'photosynthetic_radiation'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    self._save_to_cache(cache_key, df)
                    st.success(f"✓ Solar data collected: {len(df)} days")
                    return df
        except Exception as e:
            st.error(f"⚠ NASA POWER API error: {e}")
        
        return None

# ==================== FEATURE ENGINEERING CLASS ====================
class AdvancedFeatureEngineering:
    """Advanced feature engineering for time series forecasting"""
    
    @staticmethod
    def create_comprehensive_features(df, target_col='solar_power_kwh'):
        """Create comprehensive feature set"""
        df = df.copy()
        
        # Temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['date'].dt.weekday >= 5).astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Seasonal features
        seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
        df['season'] = df['month'].map(seasons)
        
        # Cyclical encoding
        for col in ['month', 'day_of_year', 'day_of_week']:
            max_val = 12 if col == 'month' else (365 if col == 'day_of_year' else 7)
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
        # Solar irradiance features
        if 'solar_irradiance_kwh_m2' in df.columns:
            df['irradiance_diff'] = df['solar_irradiance_kwh_m2'].diff()
            df['irradiance_pct_change'] = df['solar_irradiance_kwh_m2'].pct_change()
            df['irradiance_cumsum'] = df['solar_irradiance_kwh_m2'].cumsum()
        
        # Temperature features
        if all(col in df.columns for col in ['temperature_max_c', 'temperature_min_c']):
            df['temp_range'] = df['temperature_max_c'] - df['temperature_min_c']
            df['temp_mean'] = (df['temperature_max_c'] + df['temperature_min_c']) / 2
            df['temp_variability'] = df['temp_range'].rolling(7, min_periods=1).std()
        
        # Cloud features
        if 'cloud_amount' in df.columns:
            df['cloud_category'] = pd.cut(df['cloud_amount'],
                                         bins=[0, 25, 50, 75, 100],
                                         labels=[0, 1, 2, 3])
            df['cloud_category'] = df['cloud_category'].astype(float)
        
        # Lag features
        lag_periods = [1, 2, 3, 7, 14, 30]
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        rolling_windows = [3, 7, 14, 30]
        for window in rolling_windows:
            df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
            df[f'{target_col}_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
            df[f'{target_col}_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
        
        # Weather rolling features
        if 'temperature_max_c' in df.columns:
            for window in [7, 30]:
                df[f'temp_ma_{window}'] = df['temperature_max_c'].rolling(window=window, min_periods=1).mean()
        
        # Interaction features
        df = AdvancedFeatureEngineering.add_interaction_features(df)
        
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # One-hot encoding for categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df
    
    @staticmethod
    def add_interaction_features(df):
        """Add interaction features"""
        if all(col in df.columns for col in ['solar_irradiance_kwh_m2', 'temperature_max_c']):
            df['irradiance_temp_interaction'] = df['solar_irradiance_kwh_m2'] * df['temperature_max_c']
        
        if all(col in df.columns for col in ['solar_irradiance_kwh_m2', 'cloud_amount']):
            df['irradiance_cloud_interaction'] = df['solar_irradiance_kwh_m2'] * (1 - df['cloud_amount']/100)
        
        if all(col in df.columns for col in ['temperature_max_c', 'relative_humidity']):
            df['temp_humidity_interaction'] = df['temperature_max_c'] * df['relative_humidity']
        
        return df

# ==================== SOLAR POWER CALCULATION ====================
def calculate_enhanced_solar_power(df, panel_capacity_kw=1.0, efficiency=0.15, 
                                  temperature_coefficient=-0.004, soiling_loss=0.02):
    """Calculate solar power output with multiple loss factors"""
    df = df.copy()
    
    if 'solar_irradiance_kwh_m2' in df.columns:
        # Base power calculation
        df['theoretical_power'] = df['solar_irradiance_kwh_m2'] * panel_capacity_kw * efficiency
        
        # Temperature losses
        if 'temperature_max_c' in df.columns:
            temperature_loss = (df['temperature_max_c'] - 25) * temperature_coefficient
            df['temperature_adjusted_power'] = df['theoretical_power'] * (1 + temperature_loss)
        else:
            df['temperature_adjusted_power'] = df['theoretical_power']
        
        # Cloud losses
        if 'cloud_amount' in df.columns:
            cloud_loss = (df['cloud_amount'] / 100) * 0.7
            df['cloud_adjusted_power'] = df['temperature_adjusted_power'] * (1 - cloud_loss)
        else:
            df['cloud_adjusted_power'] = df['temperature_adjusted_power']
        
        # Soiling losses
        df['soiling_adjusted_power'] = df['cloud_adjusted_power'] * (1 - soiling_loss)
        
        # Wind cooling effect
        if 'wind_speed_ms' in df.columns:
            cooling_factor = np.where(df['wind_speed_ms'] > 3, 0.02, 0)
            df['final_solar_power_kwh'] = df['soiling_adjusted_power'] * (1 + cooling_factor)
        else:
            df['final_solar_power_kwh'] = df['soiling_adjusted_power']
        
        # Ensure no negative values
        df['final_solar_power_kwh'] = df['final_solar_power_kwh'].clip(lower=0)
    
    return df

# ==================== ADVANCED SOLAR MODEL CLASS ====================
class AdvancedSolarModel:
    """Advanced machine learning model ensemble for solar forecasting"""
    
    def __init__(self, config=None):
        self.config = config or {
            'rf': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 3, 'random_state': 42},
            'xgb': {'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.05, 'random_state': 42},
            'gb': {'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.1, 'random_state': 42},
            'svr': {'C': 10, 'epsilon': 0.1, 'kernel': 'rbf'},
            'lstm': {'lookback': 60, 'epochs': 100, 'batch_size': 32}
        }
        
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.feature_names = None
        
    def train_random_forest(self, X, y):
        """Train Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=self.config['rf']['n_estimators'],
            max_depth=self.config['rf']['max_depth'],
            min_samples_split=self.config['rf']['min_samples_split'],
            random_state=self.config['rf']['random_state'],
            n_jobs=-1
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model.fit(X_scaled, y)
        
        self.models['random_forest'] = model
        self.scalers['random_forest'] = scaler
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest'] = importance_df
        
        return model
    
    def train_xgboost(self, X, y):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            st.warning("⚠ XGBoost not available. Skipping XGBoost training.")
            return None
        
        model = XGBRegressor(
            n_estimators=self.config['xgb']['n_estimators'],
            max_depth=self.config['xgb']['max_depth'],
            learning_rate=self.config['xgb']['learning_rate'],
            random_state=self.config['xgb']['random_state'],
            n_jobs=-1
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model.fit(X_scaled, y)
        
        self.models['xgboost'] = model
        self.scalers['xgboost'] = scaler
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost'] = importance_df
        
        return model
    
    def train_gradient_boosting(self, X, y):
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=self.config['gb']['n_estimators'],
            max_depth=self.config['gb']['max_depth'],
            learning_rate=self.config['gb']['learning_rate'],
            random_state=self.config['gb']['random_state']
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model.fit(X_scaled, y)
        
        self.models['gradient_boosting'] = model
        self.scalers['gradient_boosting'] = scaler
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['gradient_boosting'] = importance_df
        
        return model
    
    def train_svr(self, X, y):
        """Train Support Vector Regression model"""
        model = SVR(
            C=self.config['svr']['C'],
            epsilon=self.config['svr']['epsilon'],
            kernel=self.config['svr']['kernel']
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model.fit(X_scaled, y)
        
        self.models['svr'] = model
        self.scalers['svr'] = scaler
        
        return model
    
    def train_all_models(self, X, y, model_selection=None):
        """Train all selected models"""
        if model_selection is None:
            model_selection = ['random_forest', 'xgboost', 'gradient_boosting', 'svr']
        
        self.feature_names = X.columns.tolist()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trained_models = {}
        
        for i, model_name in enumerate(model_selection):
            status_text.markdown(f"<div class='status-info'>Training {model_name.replace('_', ' ').title()}...</div>", 
                               unsafe_allow_html=True)
            
            try:
                if model_name == 'random_forest':
                    model = self.train_random_forest(X, y)
                elif model_name == 'xgboost':
                    model = self.train_xgboost(X, y)
                elif model_name == 'gradient_boosting':
                    model = self.train_gradient_boosting(X, y)
                elif model_name == 'svr':
                    model = self.train_svr(X, y)
                else:
                    continue
                
                if model is not None:
                    trained_models[model_name] = model
                    status_text.markdown(f"<div class='status-success'>✓ {model_name.replace('_', ' ').title()} trained successfully</div>", 
                                       unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠ Failed to train {model_name}: {e}")
            
            progress_bar.progress((i + 1) / len(model_selection))
            time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        return trained_models
    
    def predict(self, X, model_name='ensemble'):
        """Make predictions using specified model or ensemble"""
        if model_name == 'ensemble':
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                if name in self.scalers:
                    scaler = self.scalers[name]
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)
                    
                    predictions.append(pred)
                    
                    if name in self.performance_metrics:
                        weights.append(1 / (self.performance_metrics[name]['rmse'] + 1e-10))
                    else:
                        weights.append(1.0)
            
            if len(predictions) == 0:
                return None
            
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            weighted_predictions = np.sum(predictions * weights[:, np.newaxis], axis=0)
            return weighted_predictions
        
        elif model_name in self.models and model_name in self.scalers:
            model = self.models[model_name]
            scaler = self.scalers[model_name]
            
            X_scaled = scaler.transform(X)
            return model.predict(X_scaled)
        
        return None
    
    def evaluate_model(self, X_test, y_test, model_name='ensemble'):
        """Evaluate model performance"""
        predictions = self.predict(X_test, model_name)
        
        if predictions is None or len(predictions) != len(y_test):
            return None
        
        valid_mask = ~np.isnan(predictions)
        if not np.any(valid_mask):
            return None
        
        predictions_valid = predictions[valid_mask]
        y_test_valid = y_test.values[valid_mask] if hasattr(y_test, 'values') else y_test[valid_mask]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_valid, predictions_valid)),
            'mae': mean_absolute_error(y_test_valid, predictions_valid),
            'r2': r2_score(y_test_valid, predictions_valid),
            'mape': np.mean(np.abs((y_test_valid - predictions_valid) / (y_test_valid + 1e-10))) * 100,
            'smape': 100 * np.mean(2 * np.abs(predictions_valid - y_test_valid) /
                                  (np.abs(y_test_valid) + np.abs(predictions_valid) + 1e-10))
        }
        
        self.performance_metrics[model_name] = metrics
        return metrics
    
    def save_model(self, filepath):
        """Save model to file"""
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    @staticmethod
    def load_model(filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        model = AdvancedSolarModel(config=save_data['config'])
        model.models = save_data['models']
        model.scalers = save_data['scalers']
        model.performance_metrics = save_data['performance_metrics']
        model.feature_importance = save_data['feature_importance']
        model.feature_names = save_data['feature_names']
        
        return model, save_data['timestamp']

# ==================== MODEL PERSISTENCE ====================
class ModelPersistency:
    """Handle model saving and loading"""
    
    def __init__(self, model_dir="saved_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, model, model_name, location_name):
        """Save model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.model_dir / f"{model_name}_{location_name}_{timestamp}.pkl"
        
        model.save_model(filename)
        
        return filename
    
    def load_model(self, filename):
        """Load model from file"""
        return AdvancedSolarModel.load_model(filename)
    
    def list_saved_models(self):
        """List all saved models"""
        model_files = sorted(list(self.model_dir.glob("*.pkl")), key=os.path.getmtime, reverse=True)
        
        model_info = []
        for file in model_files:
            parts = file.stem.split('_')
            if len(parts) >= 3:
                model_info.append({
                    'file': file,
                    'name': parts[0] + '_' + parts[1] if len(parts) > 3 else parts[0],
                    'location': '_'.join(parts[2:-2]) if len(parts) > 4 else parts[2] if len(parts) > 3 else 'unknown',
                    'date': parts[-2] if len(parts) >= 2 else 'unknown',
                    'time': parts[-1] if len(parts) >= 1 else 'unknown',
                    'size': file.stat().st_size / (1024 * 1024)  # MB
                })
        
        return model_info

# ==================== VISUALIZATION FUNCTIONS ====================
def create_comparison_plot(dates, actual, predictions_dict, title="Model Predictions Comparison"):
    """Create comparison plot of actual vs predicted values"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        name='Actual',
        line=dict(color='#2E86C1', width=3),
        mode='lines'
    ))
    
    # Model predictions
    colors = ['#E74C3C', '#28B463', '#F39C12', '#8E44AD', '#17A589']
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        if len(predictions) == len(dates):
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                name=model_name.replace('_', ' ').title(),
                line=dict(color=color, width=2, dash='dot'),
                mode='lines'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Solar Power (kWh)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_metrics_comparison(metrics_dict):
    """Create metrics comparison visualization"""
    models = list(metrics_dict.keys())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                       'R² Score (Higher is Better)', 'MAPE % (Lower is Better)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # RMSE
    fig.add_trace(go.Bar(
        x=models,
        y=[metrics_dict[m]['rmse'] for m in models],
        name='RMSE',
        marker_color='#E74C3C',
        text=[f"{metrics_dict[m]['rmse']:.3f}" for m in models],
        textposition='auto'
    ), row=1, col=1)
    
    # MAE
    fig.add_trace(go.Bar(
        x=models,
        y=[metrics_dict[m]['mae'] for m in models],
        name='MAE',
        marker_color='#F39C12',
        text=[f"{metrics_dict[m]['mae']:.3f}" for m in models],
        textposition='auto'
    ), row=1, col=2)
    
    # R²
    fig.add_trace(go.Bar(
        x=models,
        y=[metrics_dict[m]['r2'] for m in models],
        name='R²',
        marker_color='#28B463',
        text=[f"{metrics_dict[m]['r2']:.3f}" for m in models],
        textposition='auto'
    ), row=2, col=1)
    
    # MAPE
    fig.add_trace(go.Bar(
        x=models,
        y=[metrics_dict[m]['mape'] for m in models],
        name='MAPE',
        marker_color='#8E44AD',
        text=[f"{metrics_dict[m]['mape']:.2f}%" for m in models],
        textposition='auto'
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white',
        title_text="Model Performance Metrics"
    )
    
    return fig

def create_feature_importance_plot(importance_df, top_n=15):
    """Create feature importance visualization"""
    importance_df = importance_df.head(top_n)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_df['importance'],
        y=importance_df['feature'],
        orientation='h',
        marker=dict(
            color=importance_df['importance'],
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"{val:.4f}" for val in importance_df['importance']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_residual_analysis(actual, predicted):
    """Create residual analysis plots"""
    residuals = actual - predicted
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Residual Distribution', 'Residual vs Predicted'),
        specs=[[{'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        name='Residuals',
        marker_color='#3498DB'
    ), row=1, col=1)
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=predicted,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='#E74C3C', size=5, opacity=0.6)
    ), row=1, col=2)
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=2)
    
    fig.update_xaxes(title_text="Residual Value", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white',
        title_text="Residual Analysis"
    )
    
    return fig

# ==================== MAIN APPLICATION ====================
def main():
    """Main application function"""
    
    # Header
    st.markdown("""
        <div class='main-header'>
            <h1 style='margin: 0; font-size: 2.5rem;'>☀️ Solar Power Forecasting System</h1>
            <p style='margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.95;'>
                Advanced Machine Learning for Rural India Solar Energy Prediction
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Model source selection
        st.subheader("Model Source")
        model_source = st.radio(
            "Choose model source:",
            ["Train New Model", "Load Saved Model"],
            help="Train a new model or load a previously saved one"
        )
        
        # Load saved model option
        if model_source == "Load Saved Model":
            model_persistence = ModelPersistency()
            saved_models = model_persistence.list_saved_models()
            
            if len(saved_models) > 0:
                st.markdown("<div class='info-box'>", unsafe_allow_html=True)
                st.write("**Available Saved Models:**")
                
                model_options = []
                for i, model_info in enumerate(saved_models):
                    label = f"{model_info['name']} - {model_info['location']} ({model_info['date']})"
                    model_options.append(label)
                    st.write(f"{i+1}. {label} - {model_info['size']:.2f} MB")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                selected_idx = st.selectbox(
                    "Select model:",
                    range(len(model_options)),
                    format_func=lambda x: model_options[x]
                )
                
                if st.button("Load Selected Model", type="primary"):
                    with st.spinner("Loading model..."):
                        try:
                            loaded_model, timestamp = model_persistence.load_model(
                                saved_models[selected_idx]['file']
                            )
                            st.session_state.trained_model = loaded_model
                            st.success(f"✓ Model loaded successfully (trained on {timestamp.strftime('%Y-%m-%d %H:%M')})")
                            
                            # Display model info
                            st.info(f"**Models available:** {', '.join(loaded_model.models.keys())}")
                        except Exception as e:
                            st.error(f"⚠ Error loading model: {e}")
            else:
                st.warning("No saved models found. Please train a new model first.")
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Location configuration
        st.subheader("📍 Location")
        location_option = st.selectbox(
            "Data Source",
            ["Predefined Location", "Custom Coordinates", "Upload Historical Data"]
        )
        
        if location_option == "Predefined Location":
            locations = {
                "Jodhpur, Rajasthan": (26.2389, 73.0243),
                "Jaisalmer, Rajasthan": (26.9157, 70.9083),
                "Anantapur, Andhra Pradesh": (14.6819, 77.6006),
                "Bikaner, Rajasthan": (28.0229, 73.3119),
                "Kurnool, Andhra Pradesh": (15.8281, 78.0373)
            }
            selected_location = st.selectbox("Select Location", list(locations.keys()))
            latitude, longitude = locations[selected_location]
            location_name = selected_location
            
        elif location_option == "Custom Coordinates":
            location_name = st.text_input("Location Name", "Custom Location")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input("Latitude", value=26.2389, format="%.4f")
            with col2:
                longitude = st.number_input("Longitude", value=73.0243, format="%.4f")
        
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            location_name = "Uploaded Data"
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Temporal parameters
        st.subheader("📅 Time Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=730))
        with col2:
            end_date = st.date_input("End Date", datetime.now() - timedelta(days=1))
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Solar system configuration
        st.subheader("🔆 Solar System")
        panel_capacity = st.number_input("Panel Capacity (kW)", value=5.0, step=0.5, min_value=0.1)
        panel_efficiency = st.slider("Panel Efficiency (%)", 10, 25, 18) / 100
        temperature_coefficient = st.slider("Temperature Coefficient (%/°C)", -0.8, 0.0, -0.4) / 100
        soiling_loss = st.slider("Soiling Loss (%)", 0, 10, 2) / 100
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        # Model selection (only for new training)
        if model_source == "Train New Model":
            st.subheader("🤖 Model Selection")
            model_options = {
                "Random Forest": "random_forest",
                "Gradient Boosting": "gradient_boosting",
                "Support Vector Regression": "svr"
            }
            
            # Add XGBoost only if available
            if XGBOOST_AVAILABLE:
                model_options_with_xgb = {"XGBoost": "xgboost"}
                model_options_with_xgb.update(model_options)
                model_options = model_options_with_xgb
                default_models = ['random_forest', 'xgboost']
            else:
                default_models = ['random_forest', 'gradient_boosting']
            
            selected_models = []
            for display_name, model_key in model_options.items():
                if st.checkbox(display_name, value=(model_key in default_models)):
                    selected_models.append(model_key)
            
            # Show info if XGBoost is not available
            if not XGBOOST_AVAILABLE:
                st.info("💡 Install XGBoost for enhanced performance: `pip install xgboost`")
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Training parameters
            st.subheader("⚡ Training Settings")
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            enable_ensemble = st.checkbox("Enable Ensemble Predictions", value=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Collection",
        "🔍 Data Exploration",
        "🎯 Model Training",
        "📈 Predictions & Analysis",
        "💡 Insights & Recommendations"
    ])
    
    # ==================== TAB 1: DATA COLLECTION ====================
    with tab1:
        st.header("📊 Data Collection & Processing")
        
        if location_option != "Upload Historical Data":
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if st.button("🚀 Collect & Process Data", type="primary", width='stretch'):
                    try:
                        collector = EnhancedSolarDataCollector()
                        
                        # Collect weather data
                        weather_df = collector.get_enhanced_weather_data(
                            latitude, longitude,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        # Collect solar data
                        solar_df = collector.get_enhanced_solar_data(
                            latitude, longitude,
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d')
                        )
                        
                        if weather_df is not None and solar_df is not None:
                            # Merge datasets
                            merged_df = pd.merge(weather_df, solar_df, on='date', how='inner')
                            merged_df['location'] = location_name
                            
                            # Calculate solar power
                            merged_df = calculate_enhanced_solar_power(
                                merged_df, panel_capacity, panel_efficiency, 
                                temperature_coefficient, soiling_loss
                            )
                            
                            # Feature engineering
                            fe = AdvancedFeatureEngineering()
                            processed_df = fe.create_comprehensive_features(
                                merged_df, 'final_solar_power_kwh'
                            )
                            
                            st.session_state.processed_data = processed_df
                            
                            st.markdown(
                                f"<div class='status-success'>✓ Successfully processed {len(processed_df):,} days of data with {len(processed_df.columns)} features</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("⚠ Failed to collect data. Please check your internet connection and try again.")
                    
                    except Exception as e:
                        st.error(f"⚠ Error during data collection: {e}")
            
            with col2:
                if st.session_state.processed_data is not None:
                    csv = st.session_state.processed_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data",
                        data=csv,
                        file_name=f"solar_data_{location_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )
            
            with col3:
                if st.session_state.processed_data is not None:
                    if st.button("🔄 Clear Data", width='stretch'):
                        st.session_state.processed_data = None
                        st.session_state.trained_model = None
                        st.session_state.model_results = None
                        st.rerun()
        
        else:
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # Calculate solar power if not present
                    if 'final_solar_power_kwh' not in df.columns:
                        df = calculate_enhanced_solar_power(
                            df, panel_capacity, panel_efficiency, 
                            temperature_coefficient, soiling_loss
                        )
                    
                    # Feature engineering
                    fe = AdvancedFeatureEngineering()
                    processed_df = fe.create_comprehensive_features(df, 'final_solar_power_kwh')
                    
                    st.session_state.processed_data = processed_df
                    
                    st.markdown(
                        f"<div class='status-success'>✓ Uploaded data processed: {len(processed_df):,} days with {len(processed_df.columns)} features</div>",
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"⚠ Error processing uploaded file: {e}")
        
        # Display data summary
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Total Days</div>
                        <div class='metric-value'>{:,}</div>
                        <div class='metric-subtitle'>Data Points</div>
                    </div>
                """.format(len(df)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Features</div>
                        <div class='metric-value'>{}</div>
                        <div class='metric-subtitle'>Engineered Variables</div>
                    </div>
                """.format(len(df.columns)), unsafe_allow_html=True)
            
            with col3:
                avg_power = df['final_solar_power_kwh'].mean()
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Avg Daily Power</div>
                        <div class='metric-value'>{:.2f}</div>
                        <div class='metric-subtitle'>kWh/day</div>
                    </div>
                """.format(avg_power), unsafe_allow_html=True)
            
            with col4:
                total_energy = df['final_solar_power_kwh'].sum()
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Total Energy</div>
                        <div class='metric-value'>{:,.0f}</div>
                        <div class='metric-subtitle'>kWh</div>
                    </div>
                """.format(total_energy), unsafe_allow_html=True)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Data preview
            st.subheader("📋 Data Preview")
            
            display_cols = ['date', 'final_solar_power_kwh', 'solar_irradiance_kwh_m2', 
                          'temperature_max_c', 'cloud_amount', 'wind_speed_ms']
            display_cols = [col for col in display_cols if col in df.columns]
            
            st.dataframe(
                df[display_cols].head(20),
                width='stretch',
                height=400
            )
    
    # ==================== TAB 2: DATA EXPLORATION ====================
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            # Time series visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.line(
                    df, x='date', y='final_solar_power_kwh',
                    title='Daily Solar Power Generation',
                    labels={'final_solar_power_kwh': 'Power (kWh)', 'date': 'Date'}
                )
                fig1.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig1, width='stretch')
            
            with col2:
                monthly_power = df.groupby(df['date'].dt.to_period('M'))['final_solar_power_kwh'].sum().reset_index()
                monthly_power['date'] = monthly_power['date'].astype(str)
                
                fig2 = px.bar(
                    monthly_power, x='date', y='final_solar_power_kwh',
                    title='Monthly Energy Generation',
                    labels={'final_solar_power_kwh': 'Energy (kWh)', 'date': 'Month'}
                )
                fig2.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig2, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Distribution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                fig3 = px.histogram(
                    df, x='final_solar_power_kwh', nbins=50,
                    title='Power Generation Distribution',
                    labels={'final_solar_power_kwh': 'Power (kWh)'}
                )
                fig3.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig3, width='stretch')
            
            with col2:
                seasonal_power = df.groupby('season')['final_solar_power_kwh'].mean().reset_index()
                season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
                seasonal_power['season_name'] = seasonal_power['season'].map(season_names)
                
                fig4 = px.bar(
                    seasonal_power, x='season_name', y='final_solar_power_kwh',
                    title='Average Power by Season',
                    labels={'final_solar_power_kwh': 'Avg Power (kWh)', 'season_name': 'Season'},
                    color='final_solar_power_kwh',
                    color_continuous_scale='Viridis'
                )
                fig4.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig4, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Correlation analysis
            st.subheader("🔗 Feature Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'final_solar_power_kwh' in numeric_cols:
                corr_matrix = df[numeric_cols].corr()
                top_correlations = corr_matrix['final_solar_power_kwh'].sort_values(ascending=False)[1:16]
                
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(
                    x=top_correlations.values,
                    y=top_correlations.index,
                    orientation='h',
                    marker=dict(
                        color=top_correlations.values,
                        colorscale='RdYlGn',
                        showscale=True,
                        cmin=-1,
                        cmax=1
                    ),
                    text=[f"{val:.3f}" for val in top_correlations.values],
                    textposition='auto'
                ))
                
                fig5.update_layout(
                    title="Top 15 Features Correlated with Solar Power",
                    xaxis_title="Correlation Coefficient",
                    yaxis_title="Feature",
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig5, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            
            summary_cols = ['final_solar_power_kwh', 'solar_irradiance_kwh_m2',
                          'temperature_max_c', 'cloud_amount', 'wind_speed_ms']
            summary_cols = [col for col in summary_cols if col in df.columns]
            
            st.dataframe(
                df[summary_cols].describe().round(3),
                width='stretch'
            )
        
        else:
            st.info("👈 Please collect or upload data first in the 'Data Collection' tab")
    
    # ==================== TAB 3: MODEL TRAINING ====================
    with tab3:
        st.header("🎯 Model Training & Validation")
        
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            if 'final_solar_power_kwh' in df.columns:
                # Prepare data
                exclude_cols = ['date', 'location', 'final_solar_power_kwh']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                X = df[feature_cols]
                y = df['final_solar_power_kwh']
                
                # Split data
                if model_source == "Train New Model":
                    split_idx = int(len(X) * (1 - test_size))
                else:
                    test_size_default = 0.20
                    split_idx = int(len(X) * (1 - test_size_default))
                
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                dates_train, dates_test = df['date'][:split_idx], df['date'][split_idx:]
                
                # Display split info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                        <div class='info-box'>
                            <strong>Training Samples</strong><br>
                            <span style='font-size: 1.5rem; color: #667eea;'>{:,}</span>
                        </div>
                    """.format(len(X_train)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class='info-box'>
                            <strong>Test Samples</strong><br>
                            <span style='font-size: 1.5rem; color: #667eea;'>{:,}</span>
                        </div>
                    """.format(len(X_test)), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class='info-box'>
                            <strong>Features</strong><br>
                            <span style='font-size: 1.5rem; color: #667eea;'>{}</span>
                        </div>
                    """.format(len(feature_cols)), unsafe_allow_html=True)
                
                st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                
                # Training buttons
                if model_source == "Train New Model":
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if st.button("🚀 Train Models", type="primary", width='stretch'):
                            if len(selected_models) == 0:
                                st.warning("⚠ Please select at least one model to train")
                            else:
                                try:
                                    advanced_model = AdvancedSolarModel()
                                    trained_models = advanced_model.train_all_models(
                                        X_train, y_train, selected_models
                                    )
                                    
                                    st.session_state.trained_model = advanced_model
                                    st.session_state.X_test = X_test
                                    st.session_state.y_test = y_test
                                    st.session_state.dates_test = dates_test
                                    
                                    st.markdown(
                                        f"<div class='status-success'>✓ Successfully trained {len(trained_models)} models</div>",
                                        unsafe_allow_html=True
                                    )
                                except Exception as e:
                                    st.error(f"⚠ Training error: {e}")
                    
                    with col2:
                        if st.session_state.trained_model is not None:
                            model_persistence = ModelPersistency()
                            if st.button("💾 Save Model", width='stretch'):
                                try:
                                    filename = model_persistence.save_model(
                                        st.session_state.trained_model,
                                        "solar_model",
                                        location_name.replace(' ', '_').replace(',', '')
                                    )
                                    st.success(f"✓ Model saved: {filename.name}")
                                except Exception as e:
                                    st.error(f"⚠ Save error: {e}")
                
                # Evaluate models
                if st.session_state.trained_model is not None:
                    model = st.session_state.trained_model
                    
                    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                    st.subheader("📊 Model Performance")
                    
                    # Calculate metrics for all models
                    metrics_results = {}
                    predictions_dict = {}
                    
                    for model_name in model.models.keys():
                        pred = model.predict(X_test, model_name)
                        if pred is not None and len(pred) == len(y_test):
                            predictions_dict[model_name] = pred
                            metrics = model.evaluate_model(X_test, y_test, model_name)
                            if metrics:
                                metrics_results[model_name] = metrics
                    
                    # Add ensemble if enabled
                    if model_source == "Train New Model" and enable_ensemble and len(model.models) > 1:
                        ensemble_pred = model.predict(X_test, 'ensemble')
                        if ensemble_pred is not None:
                            predictions_dict['ensemble'] = ensemble_pred
                            metrics = model.evaluate_model(X_test, y_test, 'ensemble')
                            if metrics:
                                metrics_results['ensemble'] = metrics
                    
                    if metrics_results:
                        # Save results
                        st.session_state.model_results = {
                            'predictions': predictions_dict,
                            'metrics': metrics_results,
                            'dates': dates_test
                        }
                        
                        # Display metrics table
                        metrics_df = pd.DataFrame(metrics_results).T
                        
                        st.dataframe(
                            metrics_df.style.format({
                                'rmse': '{:.4f}',
                                'mae': '{:.4f}',
                                'r2': '{:.4f}',
                                'mape': '{:.2f}%',
                                'smape': '{:.2f}%'
                            }).background_gradient(cmap='RdYlGn_r', subset=['rmse', 'mae', 'mape', 'smape'])
                              .background_gradient(cmap='RdYlGn', subset=['r2']),
                            width='stretch'
                        )
                        
                        # Best model
                        best_model = metrics_df['r2'].idxmax()
                        best_r2 = metrics_df.loc[best_model, 'r2']
                        best_rmse = metrics_df.loc[best_model, 'rmse']
                        
                        st.markdown(f"""
                            <div class='status-success'>
                                <strong>🏆 Best Model: {best_model.replace('_', ' ').title()}</strong><br>
                                R² Score: {best_r2:.4f} | RMSE: {best_rmse:.4f}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics visualization
                        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                        fig_metrics = create_metrics_comparison(metrics_results)
                        st.plotly_chart(fig_metrics, width='stretch')
                        
                        # Feature importance
                        if 'random_forest' in model.feature_importance:
                            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
                            st.subheader("🎯 Feature Importance")
                            
                            importance_df = model.feature_importance['random_forest']
                            fig_importance = create_feature_importance_plot(importance_df, top_n=20)
                            st.plotly_chart(fig_importance, width='stretch')
                
                else:
                    if model_source == "Train New Model":
                        st.info("👆 Click 'Train Models' to start training")
                    else:
                        st.info("👈 Load a saved model from the sidebar to view performance")
        
        else:
            st.info("👈 Please collect or upload data first in the 'Data Collection' tab")
    
    # ==================== TAB 4: PREDICTIONS & ANALYSIS ====================
    with tab4:
        st.header("📈 Predictions & Analysis")
        
        if st.session_state.model_results is not None:
            results = st.session_state.model_results
            predictions_dict = results['predictions']
            metrics_dict = results['metrics']
            dates_test = results['dates']
            y_test = st.session_state.y_test
            
            # Model comparison plot
            st.subheader("📊 Actual vs Predicted Values")
            
            actual_values = y_test.values if hasattr(y_test, 'values') else y_test
            fig_comparison = create_comparison_plot(
                dates_test.values,
                actual_values,
                predictions_dict,
                "Model Predictions Comparison"
            )
            st.plotly_chart(fig_comparison, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Residual analysis
            st.subheader("🔍 Residual Analysis")
            
            # Use ensemble if available, otherwise use first model
            pred_key = 'ensemble' if 'ensemble' in predictions_dict else list(predictions_dict.keys())[0]
            predictions = predictions_dict[pred_key]
            
            fig_residual = create_residual_analysis(actual_values, predictions)
            st.plotly_chart(fig_residual, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Error metrics by time period
            st.subheader("📅 Error Analysis by Period")
            
            results_df = pd.DataFrame({
                'date': dates_test.values,
                'actual': actual_values,
                'predicted': predictions,
                'error': actual_values - predictions,
                'absolute_error': np.abs(actual_values - predictions)
            })
            results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M').astype(str)
            
            monthly_errors = results_df.groupby('month').agg({
                'error': 'mean',
                'absolute_error': 'mean',
                'actual': 'mean',
                'predicted': 'mean'
            }).reset_index()
            
            fig_monthly = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Mean Error by Month', 'Mean Absolute Error by Month')
            )
            
            fig_monthly.add_trace(go.Bar(
                x=monthly_errors['month'],
                y=monthly_errors['error'],
                name='Mean Error',
                marker_color='#3498DB'
            ), row=1, col=1)
            
            fig_monthly.add_trace(go.Bar(
                x=monthly_errors['month'],
                y=monthly_errors['absolute_error'],
                name='MAE',
                marker_color='#E74C3C'
            ), row=1, col=2)
            
            fig_monthly.update_layout(
                height=400,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig_monthly, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Download predictions
            st.subheader("💾 Export Predictions")
            
            export_df = pd.DataFrame({
                'date': dates_test.values,
                'actual_power_kwh': actual_values
            })
            
            for model_name, pred in predictions_dict.items():
                if len(pred) == len(export_df):
                    export_df[f'{model_name}_prediction_kwh'] = pred
                    export_df[f'{model_name}_error_kwh'] = actual_values - pred
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.dataframe(export_df.head(20), width='stretch')
            
            with col2:
                csv_export = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_export,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
        
        else:
            st.info("👈 Please train or load a model first in the 'Model Training' tab")
    
    # ==================== TAB 5: INSIGHTS & RECOMMENDATIONS ====================
    with tab5:
        st.header("💡 System Insights & Recommendations")
        
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            # Energy production analysis
            st.subheader("⚡ Energy Production Analysis")
            
            yearly_energy = df.groupby(df['date'].dt.year)['final_solar_power_kwh'].sum()
            monthly_avg = df.groupby(df['date'].dt.month)['final_solar_power_kwh'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_yearly = px.bar(
                    x=yearly_energy.index.astype(str),
                    y=yearly_energy.values,
                    title="Annual Energy Production",
                    labels={'x': 'Year', 'y': 'Energy (kWh)'},
                    color=yearly_energy.values,
                    color_continuous_scale='Viridis'
                )
                fig_yearly.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig_yearly, width='stretch')
            
            with col2:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig_monthly_pattern = px.line(
                    x=[month_names[i-1] for i in monthly_avg.index],
                    y=monthly_avg.values,
                    title="Average Monthly Production Pattern",
                    labels={'x': 'Month', 'y': 'Average Power (kWh)'},
                    markers=True
                )
                fig_monthly_pattern.update_layout(template='plotly_white', height=400)
                fig_monthly_pattern.update_traces(line_color='#E74C3C', marker=dict(size=8))
                st.plotly_chart(fig_monthly_pattern, width='stretch')
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Performance metrics
            st.subheader("📊 System Performance Metrics")
            
            capacity_factor = (df['final_solar_power_kwh'].mean() / (panel_capacity * 24)) * 100
            
            if 'solar_irradiance_kwh_m2' in df.columns:
                performance_ratio = (df['final_solar_power_kwh'].sum() / 
                                   (df['solar_irradiance_kwh_m2'].sum() * panel_capacity * panel_efficiency)) * 100
            else:
                performance_ratio = 0
            
            annual_energy = df['final_solar_power_kwh'].sum() / (len(df) / 365)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Capacity Factor</div>
                        <div class='metric-value'>{:.1f}%</div>
                        <div class='metric-subtitle'>System Utilization</div>
                    </div>
                """.format(capacity_factor), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Performance Ratio</div>
                        <div class='metric-value'>{:.1f}%</div>
                        <div class='metric-subtitle'>vs Theoretical</div>
                    </div>
                """.format(performance_ratio), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Annual Energy</div>
                        <div class='metric-value'>{:,.0f}</div>
                        <div class='metric-subtitle'>kWh/year</div>
                    </div>
                """.format(annual_energy), unsafe_allow_html=True)
            
            with col4:
                peak_power = df['final_solar_power_kwh'].max()
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Peak Power</div>
                        <div class='metric-value'>{:.2f}</div>
                        <div class='metric-subtitle'>kWh (daily)</div>
                    </div>
                """.format(peak_power), unsafe_allow_html=True)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("💡 System Optimization Recommendations")
            
            recommendations = []
            
            # Capacity factor check
            if capacity_factor < 15:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Low Capacity Factor',
                    'message': 'Consider optimizing panel orientation and tilt angle to improve capacity factor above 15%.'
                })
            elif capacity_factor > 20:
                recommendations.append({
                    'type': 'success',
                    'title': 'Excellent Capacity Factor',
                    'message': f'Your system is performing well with a {capacity_factor:.1f}% capacity factor.'
                })
            
            # Performance ratio check
            if performance_ratio < 75 and performance_ratio > 0:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Below Optimal Performance',
                    'message': 'Performance ratio is below 75%. Check for shading, soiling, or panel degradation.'
                })
            elif performance_ratio >= 80:
                recommendations.append({
                    'type': 'success',
                    'title': 'High Performance Ratio',
                    'message': f'Excellent performance ratio of {performance_ratio:.1f}%. System is well-maintained.'
                })
            
            # Seasonal analysis
            best_month = monthly_avg.idxmax()
            worst_month = monthly_avg.idxmin()
            month_names_dict = {i+1: name for i, name in enumerate(month_names)}
            
            recommendations.append({
                'type': 'info',
                'title': 'Seasonal Patterns',
                'message': f'Best production: {month_names_dict[best_month]} ({monthly_avg[best_month]:.2f} kWh/day). '
                          f'Lowest: {month_names_dict[worst_month]} ({monthly_avg[worst_month]:.2f} kWh/day). '
                          f'Plan maintenance during low-production months.'
            })
            
            # Monsoon impact
            if worst_month in [6, 7, 8]:
                recommendations.append({
                    'type': 'info',
                    'title': 'Monsoon Impact',
                    'message': 'Monsoon season shows reduced output. Schedule system maintenance and cleaning before peak solar months.'
                })
            
            # Display recommendations
            for rec in recommendations:
                if rec['type'] == 'success':
                    st.markdown(f"""
                        <div class='status-success'>
                            <strong>{rec['title']}</strong><br>
                            {rec['message']}
                        </div>
                    """, unsafe_allow_html=True)
                elif rec['type'] == 'warning':
                    st.markdown(f"""
                        <div class='status-warning'>
                            <strong>⚠ {rec['title']}</strong><br>
                            {rec['message']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='status-info'>
                            <strong>ℹ {rec['title']}</strong><br>
                            {rec['message']}
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Economic analysis
            st.subheader("💰 Economic Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                electricity_rate = st.number_input(
                    "Electricity Rate (₹/kWh)",
                    value=6.0,
                    min_value=0.0,
                    step=0.5,
                    help="Average electricity tariff in your region"
                )
            
            with col2:
                system_cost_per_kw = st.number_input(
                    "System Cost (₹/kW)",
                    value=60000,
                    min_value=0,
                    step=5000,
                    help="Total installed cost per kW"
                )
            
            # Calculate economics
            annual_savings = annual_energy * electricity_rate
            system_cost = panel_capacity * system_cost_per_kw
            
            if annual_savings > 0:
                payback_period = system_cost / annual_savings
                roi_25_years = ((annual_savings * 25) - system_cost) / system_cost * 100
            else:
                payback_period = float('inf')
                roi_25_years = 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Annual Savings</div>
                        <div class='metric-value'>₹{:,.0f}</div>
                        <div class='metric-subtitle'>per year</div>
                    </div>
                """.format(annual_savings), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>System Cost</div>
                        <div class='metric-value'>₹{:,.0f}</div>
                        <div class='metric-subtitle'>Total Investment</div>
                    </div>
                """.format(system_cost), unsafe_allow_html=True)
            
            with col3:
                payback_text = f"{payback_period:.1f}" if payback_period != float('inf') else "N/A"
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>Payback Period</div>
                        <div class='metric-value'>{}</div>
                        <div class='metric-subtitle'>years</div>
                    </div>
                """.format(payback_text), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-title'>25-Year ROI</div>
                        <div class='metric-value'>{:.0f}%</div>
                        <div class='metric-subtitle'>Return on Investment</div>
                    </div>
                """.format(roi_25_years), unsafe_allow_html=True)
            
            # ROI visualization
            if payback_period != float('inf'):
                years = np.arange(0, 26)
                cumulative_savings = annual_savings * years - system_cost
                
                fig_roi = go.Figure()
                
                fig_roi.add_trace(go.Scatter(
                    x=years,
                    y=cumulative_savings,
                    mode='lines',
                    name='Cumulative Savings',
                    line=dict(color='#28B463', width=3),
                    fill='tozeroy'
                ))
                
                fig_roi.add_hline(y=0, line_dash="dash", line_color="black")
                
                fig_roi.update_layout(
                    title="25-Year Cumulative Financial Benefit",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Savings (₹)",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_roi, width='stretch')
        
        else:
            st.info("👈 Please collect or upload data first in the 'Data Collection' tab")
    
    # Footer
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p>Solar Power Forecasting System v2.0 | Enhanced with Advanced ML & Economic Analysis</p>
            <p style='font-size: 0.9rem;'>Developed for Rural India Solar Energy Optimization</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()