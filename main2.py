"""
Solar Power Forecasting Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import time

st.set_page_config(
    page_title="Solar Power Forecasting",
    page_icon="sun",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f9fdf7 0%, #f5faf3 100%);
        padding: 1rem 2rem;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fdf7 100%);
        border-right: 1px solid #e8f5e9;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        text-align: center;
        padding: 15px;
        margin-bottom: 20px;
        letter-spacing: -0.5px;
        border-bottom: 3px solid #f9a825;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #1b5e20;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #c8e6c9;
        letter-spacing: -0.3px;
    }
    
    .stButton > button {
        background: #2e7d32;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(46, 125, 50, 0.2);
    }
    
    .stButton > button:hover {
        background: #1b5e20;
        box-shadow: 0 4px 8px rgba(46, 125, 50, 0.3);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f9fdf7;
        border-radius: 10px;
        padding: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2e7d32;
        color: white;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2e7d32;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        color: #1b5e20;
    }
    
    .stSlider > div > div > div {
        background-color: #2e7d32;
    }
    
    .stDownloadButton > button {
        background: #f9a825;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: #f57f17;
        transform: translateY(-2px);
    }
    
    .stProgress > div > div {
        background: #2e7d32;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1b5e20;
        font-weight: 700;
    }
    
    p, span, label {
        color: #424242;
    }
    </style>
    """, unsafe_allow_html=True)


class SolarDataCollector:
    def __init__(self):
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def get_solar_data_nasa_power(self, latitude, longitude, start_date, end_date):
        today = datetime.now().date()
        max_end_date = today
        
        if end_date > max_end_date:
            end_date = max_end_date
        
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
            st.warning(f"API Error: {str(e)[:100]}...")
            return None
    
    def generate_smart_synthetic_data(self, latitude, longitude, start_date, end_date):
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        if latitude > 20:
            base_temp = 25 + 12 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 5.5 + 2.5 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        else:
            base_temp = 28 + 8 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 6 + 2 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        
        np.random.seed(int(latitude * longitude))
        
        solar_irradiance = np.maximum(base_irradiance + np.random.normal(0, 0.8, len(date_range)), 1.5)
        
        df = pd.DataFrame({
            'date': date_range,
            'solar_irradiance_kwh_m2': np.abs(solar_irradiance),
            'temperature_c': base_temp + np.random.normal(0, 3, len(date_range)),
            'relative_humidity': np.clip(50 + 25 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + np.random.normal(0, 12, len(date_range)), 20, 100),
            'cloudcover_mean': np.clip(40 + 35 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + np.random.normal(0, 18, len(date_range)), 0, 100)
        })
        
        df['temperature_2m_mean'] = df['temperature_c']
        df['clear_sky_irradiance_kwh_m2'] = np.abs(df['solar_irradiance_kwh_m2'] * (1.2 + 0.3 * np.random.random(len(date_range))))
        df['wind_speed_ms'] = np.abs(4 + 2 * np.random.random(len(date_range)))
        
        df['solar_irradiance_kwh_m2'] = np.abs(df['solar_irradiance_kwh_m2']).clip(1.0, 8)
        df['temperature_2m_mean'] = df['temperature_2m_mean'].clip(10, 45)
        df['relative_humidity'] = df['relative_humidity'].clip(20, 100)
        df['cloudcover_mean'] = df['cloudcover_mean'].clip(0, 100)
        df['clear_sky_irradiance_kwh_m2'] = np.abs(df['clear_sky_irradiance_kwh_m2']).clip(1.5, 10)
        df['wind_speed_ms'] = np.abs(df['wind_speed_ms']).clip(0.5, 15)
        
        return df
    
    def collect_data_for_location(self, location_name, latitude, longitude, start_date, end_date):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Fetching solar data from NASA POWER...")
        progress_bar.progress(30)
        
        solar_df = self.get_solar_data_nasa_power(latitude, longitude, start_date, end_date)
        
        if solar_df is not None and not solar_df.empty:
            status_text.text("Finalizing data collection...")
            progress_bar.progress(90)
            merged_df = solar_df.copy()
            if 'cloudcover_mean' not in merged_df.columns:
                merged_df['cloudcover_mean'] = 50
            if 'temperature_2m_mean' not in merged_df.columns and 'temperature_c' in merged_df.columns:
                merged_df['temperature_2m_mean'] = merged_df['temperature_c']
        else:
            status_text.text("Generating synthetic data...")
            progress_bar.progress(70)
            merged_df = self.generate_smart_synthetic_data(latitude, longitude, start_date, end_date)
            st.info(f"Using synthetic data for {location_name}")
        
        for col in ['solar_irradiance_kwh_m2', 'clear_sky_irradiance_kwh_m2']:
            if col in merged_df.columns:
                merged_df[col] = np.abs(merged_df[col])
        
        merged_df['location'] = location_name
        merged_df['latitude'] = latitude
        merged_df['longitude'] = longitude
        
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
        
        progress_bar.progress(100)
        status_text.text(f"Data collection complete: {len(merged_df)} days")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()
        
        return merged_df


def calculate_solar_power(df, panel_capacity_kw=5.0, efficiency=0.17):
    df = df.copy()
    
    if 'solar_irradiance_kwh_m2' not in df.columns:
        st.warning("No irradiance data found.")
        return df
    
    panel_area = panel_capacity_kw / (1.0 * efficiency)
    df['solar_power_kwh'] = df['solar_irradiance_kwh_m2'] * panel_area * efficiency
    
    if 'clearness_index' in df.columns:
        df['solar_power_kwh'] *= df['clearness_index'].clip(0.5, 0.9)
    
    if 'temp_efficiency_factor' in df.columns:
        df['solar_power_kwh'] *= df['temp_efficiency_factor'].clip(0.85, 1.05)
    
    if 'cloud_reduction_factor' in df.columns:
        df['solar_power_kwh'] *= df['cloud_reduction_factor'].clip(0.4, 1.0)
    
    max_theoretical = panel_capacity_kw * 8
    df['solar_power_kwh'] = df['solar_power_kwh'].clip(lower=0.1, upper=max_theoretical)
    
    return df


class SolarFeatureEngineering:
    @staticmethod
    def add_temporal_features(df):
        df = df.copy()
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek
        
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
        df = df.copy()
        
        if 'solar_irradiance_kwh_m2' in df.columns and 'clear_sky_irradiance_kwh_m2' in df.columns:
            df['clearness_index'] = df['solar_irradiance_kwh_m2'] / (df['clear_sky_irradiance_kwh_m2'] + 0.001)
            df['clearness_index'] = df['clearness_index'].clip(0, 1)
        
        temp_col = None
        for col in ['temperature_2m_mean', 'temperature_c']:
            if col in df.columns:
                temp_col = col
                break
        
        if temp_col is not None:
            df['temp_efficiency_factor'] = np.maximum(0.85, 1.0 - 0.003 * np.abs(df[temp_col] - 25))
            df['temp_efficiency_factor'] = df['temp_efficiency_factor'].clip(0.85, 1.05)
        
        if 'cloudcover_mean' in df.columns:
            df['cloud_reduction_factor'] = 1.0 - (df['cloudcover_mean'] / 100) * 0.6
            df['cloud_reduction_factor'] = df['cloud_reduction_factor'].clip(0.4, 1.0)
        
        return df


class HybridForecastingModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=200, random_state=42):
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
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=1000,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1
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
            st.warning(f"ARIMA failed: {str(e)}")
            return None, None
    
    def train_sarima(self, y_train, y_test, order=(1,1,1), seasonal_order=(1,1,1,12)):
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
            st.warning(f"SARIMA failed: {str(e)}")
            return None, None
    
    def train_exponential_smoothing(self, y_train, y_test, seasonal_periods=12):
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
            st.warning(f"Exponential Smoothing failed: {str(e)}")
            return None, None
    
    def train_ensemble_model(self, X_train, y_train, X_test, y_test):
        if 'Random Forest' not in self.models:
            self.train_random_forest(X_train, y_train, X_test, y_test)
        if 'Gradient Boosting' not in self.models:
            self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        
        rf_scaler = self.scalers['Random Forest']
        gb_scaler = self.scalers['Gradient Boosting']
        
        X_test_scaled_rf = rf_scaler.transform(X_test)
        X_test_scaled_gb = gb_scaler.transform(X_test)
        
        rf_pred = self.models['Random Forest'].predict(X_test_scaled_rf)
        gb_pred = self.models['Gradient Boosting'].predict(X_test_scaled_gb)
        
        ensemble_pred = (0.6 * rf_pred + 0.4 * gb_pred)
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        self.models['Ensemble (RF+GB)'] = {
            'rf': self.models['Random Forest'],
            'gb': self.models['Gradient Boosting'],
            'weights': [0.6, 0.4]
        }
        
        return metrics, ensemble_pred
    
    def _calculate_metrics(self, y_true, y_pred):
        if y_pred is None or len(y_pred) == 0:
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
        
        metrics['RMSE'] = float(f"{metrics['RMSE']:.3f}")
        metrics['MAE'] = float(f"{metrics['MAE']:.3f}")
        metrics['R2'] = float(f"{metrics['R2']:.3f}")
        metrics['MAPE'] = float(f"{metrics['MAPE']:.1f}")
        
        return metrics


def plot_historical_data(df, location_name):
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Solar Power Generation',),
    )
    
    if 'solar_power_kwh' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['solar_power_kwh'],
                name='Solar Power', 
                line=dict(color='#2e7d32', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(46, 125, 50, 0.1)'
            ),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=1, col=1, gridcolor='#e8f5e9')
    fig.update_yaxes(title_text="kWh", row=1, col=1, gridcolor='#e8f5e9')
    
    fig.update_layout(
        height=400,
        title={
            'text': f"Historical Data - {location_name}",
            'font': {'size': 16, 'color': '#1b5e20'}
        },
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f9fdf7',
        font=dict(size=11, color='#424242'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def plot_predictions_vs_actual(y_test, predictions_dict, test_dates):
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test,
            name='Actual',
            line=dict(color='#1b5e20', width=3),
            mode='lines'
        )
    )
    
    colors = ['#2e7d32', '#558b2f', '#4caf50', '#f9a825', '#c62828', '#00695c']
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=y_pred,
                name=model_name,
                line=dict(color=colors[idx % len(colors)], width=2, dash='dot'),
                mode='lines'
            )
        )
    
    fig.update_layout(
        title={
            'text': 'Model Predictions vs Actual Values',
            'font': {'size': 16, 'color': '#1b5e20'}
        },
        xaxis_title='Date',
        yaxis_title='Solar Power (kWh)',
        height=400,
        hovermode='x unified',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f9fdf7',
        font=dict(size=11, color='#424242'),
        xaxis=dict(gridcolor='#e8f5e9'),
        yaxis=dict(gridcolor='#e8f5e9'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig


def main():
    st.markdown('<p class="main-header">Solar Power Forecasting Dashboard</p>',
                unsafe_allow_html=True)
    
    st.sidebar.title("Settings")
    
    st.sidebar.markdown("**Location**")
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
        "Select a location",
        list(RURAL_LOCATIONS.keys()),
        index=0
    )
    
    if location_choice == 'Custom Location':
        location_name = st.sidebar.text_input("Location Name", "")
        latitude = st.sidebar.number_input("Latitude", format="%.4f")
        longitude = st.sidebar.number_input("Longitude", format="%.4f")
    else:
        location_name = location_choice
        latitude, longitude = RURAL_LOCATIONS[location_choice]
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Period**")
    
    today = datetime.now().date()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=365)
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date = end_date - timedelta(days=30)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Solar System**")
    panel_capacity = st.sidebar.slider("Panel Capacity (kW)", 1.0, 100.0, 5.0, 0.5)
    panel_efficiency = st.sidebar.slider("Panel Efficiency (%)", 10, 25, 17) / 100
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models to Train**")
    available_models = ['Random Forest', 'Gradient Boosting', 'Linear Regression', 
                       'SVR', 'Neural Network', 'Ensemble (RF+GB)']
    
    if STATSMODELS_AVAILABLE:
        available_models.extend(['ARIMA', 'SARIMA', 'Exponential Smoothing'])
    
    selected_models = st.sidebar.multiselect(
        "Select forecasting models",
        available_models,
        default=['Random Forest', 'Gradient Boosting', 'Ensemble (RF+GB)']
    )
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = HybridForecastingModels()
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = pd.DataFrame()
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'test_dates' not in st.session_state:
        st.session_state.test_dates = None
    
    tab1, tab2 = st.tabs(["Data Collection", "Model Training"])
    
    with tab1:
        st.markdown('<p class="sub-header">Data Collection and Analysis</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Location", location_name)
        with col2:
            days_diff = (end_date - start_date).days
            st.metric("Period", f"{days_diff} days")
        with col3:
            st.metric("System Capacity", f"{panel_capacity} kW")
        
        if st.button("Collect Data", type="primary", use_container_width=True):
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
                        
                        st.success(f"Data collected: {len(df)} days")
                        
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Total Days", len(df))
                        with metrics_col2:
                            if 'solar_power_kwh' in df.columns:
                                avg_power = df['solar_power_kwh'].mean()
                                st.metric("Avg Daily Power", f"{avg_power:.1f} kWh")
                        with metrics_col3:
                            if 'solar_power_kwh' in df.columns:
                                total_power = df['solar_power_kwh'].sum()
                                st.metric("Total Energy", f"{total_power:.0f} kWh")
                        
                    else:
                        st.error("Failed to collect data.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("---")
            st.markdown("### Historical Data")
            
            fig_historical = plot_historical_data(df, location_name)
            st.plotly_chart(fig_historical, use_container_width=True)
            
            with st.expander("View Data Table"):
                display_cols = ['date', 'solar_power_kwh']
                
                available_cols = [col for col in display_cols if col in df.columns]
                if len(available_cols) > 0:
                    st.dataframe(df[available_cols].head(50), use_container_width=True)
            
            st.markdown("### Data Summary")
            if 'solar_power_kwh' in df.columns:
                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.metric("Mean Daily Power", f"{df['solar_power_kwh'].mean():.2f} kWh")
                    st.metric("Max Daily Power", f"{df['solar_power_kwh'].max():.2f} kWh")
                with summary_col2:
                    st.metric("Std Deviation", f"{df['solar_power_kwh'].std():.2f} kWh")
                    st.metric("Median Daily Power", f"{df['solar_power_kwh'].median():.2f} kWh")
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"solar_data_{location_name}_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Click Collect Data to start")
    
    with tab2:
        st.markdown('<p class="sub-header">Model Training and Evaluation</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            if 'solar_power_kwh' not in df.columns:
                st.error("Solar power data not available.")
            else:
                st.info(f"Training on {len(df)} days of data")
                
                test_size = 20
                random_state = 42
                
                if st.button("Train Models", type="primary", use_container_width=True):
                    if not selected_models:
                        st.error("Please select at least one model")
                    else:
                        with st.spinner(f"Training {len(selected_models)} models..."):
                            try:
                                fe = SolarFeatureEngineering()
                                df_features = fe.add_temporal_features(df)
                                df_features = fe.add_solar_features(df_features)
                                
                                exclude_cols = ['date', 'location', 'latitude', 'longitude', 
                                              'solar_power_kwh']
                                
                                feature_cols = []
                                for col in df_features.columns:
                                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col]):
                                        feature_cols.append(col)
                                
                                X = df_features[feature_cols]
                                y = df_features['solar_power_kwh']
                                
                                X = X.fillna(X.mean())
                                y = y.fillna(y.mean())
                                
                                split_idx = int(len(X) * (1 - test_size/100))
                                X_train, X_test = X[:split_idx], X[split_idx:]
                                y_train, y_test = y[:split_idx], y[split_idx:]
                                
                                st.session_state.y_test = y_test
                                st.session_state.test_dates = df_features['date'].iloc[split_idx:].values
                                
                                st.write(f"Training set: {len(X_train)} days | Test set: {len(X_test)} days")
                                
                                forecaster = st.session_state.forecaster
                                all_metrics = []
                                predictions_dict = {}
                                
                                for model_name in selected_models:
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
                                            predictions_dict[model_name] = y_pred
                                            st.success(f"{model_name} trained")
                                        else:
                                            st.warning(f"{model_name} completed with warnings")
                                            
                                    except Exception as e:
                                        st.error(f"{model_name} failed: {str(e)[:100]}")
                                
                                if all_metrics:
                                    metrics_df = pd.concat(all_metrics, ignore_index=True)
                                    st.session_state.model_metrics = metrics_df
                                    st.session_state.predictions = predictions_dict
                                    
                                    st.success(f"Successfully trained {len(all_metrics)} models")
                                    
                                    st.markdown("---")
                                    
                                    st.markdown("### Predictions vs Actual")
                                    fig_predictions = plot_predictions_vs_actual(
                                        y_test, 
                                        predictions_dict, 
                                        st.session_state.test_dates
                                    )
                                    st.plotly_chart(fig_predictions, use_container_width=True)
                                    
                                    st.markdown("---")
                                    
                                    st.markdown("### Performance Metrics")
                                    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
                                    
                                    if not metrics_df.empty:
                                        best_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin()]
                                        best_r2 = metrics_df.loc[metrics_df['R2'].idxmax()]
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Best RMSE", f"{best_rmse['Model']}")
                                            st.write(f"Value: {best_rmse['RMSE']:.3f}")
                                        with col2:
                                            st.metric("Best R2", f"{best_r2['Model']}")
                                            st.write(f"Value: {best_r2['R2']:.3f}")
                                        with col3:
                                            avg_mape = metrics_df['MAPE'].mean()
                                            st.metric("Average MAPE", f"{avg_mape:.1f}%")
                                
                            except Exception as e:
                                st.error(f"Training failed: {str(e)}")
                
                elif st.session_state.model_metrics is not None and not st.session_state.model_metrics.empty:
                    st.info("Displaying previous results")
                    
                    st.markdown("---")
                    
                    if st.session_state.predictions and st.session_state.y_test is not None:
                        st.markdown("### Predictions vs Actual")
                        fig_predictions = plot_predictions_vs_actual(
                            st.session_state.y_test, 
                            st.session_state.predictions, 
                            st.session_state.test_dates
                        )
                        st.plotly_chart(fig_predictions, use_container_width=True)
                        
                        st.markdown("---")
                    
                    st.markdown("### Performance Metrics")
                    st.dataframe(st.session_state.model_metrics.set_index('Model'), use_container_width=True)
                    
                    metrics_df = st.session_state.model_metrics
                    if not metrics_df.empty:
                        best_rmse = metrics_df.loc[metrics_df['RMSE'].idxmin()]
                        best_r2 = metrics_df.loc[metrics_df['R2'].idxmax()]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best RMSE", f"{best_rmse['Model']}")
                            st.write(f"Value: {best_rmse['RMSE']:.3f}")
                        with col2:
                            st.metric("Best R2", f"{best_r2['Model']}")
                            st.write(f"Value: {best_r2['R2']:.3f}")
                        with col3:
                            avg_mape = metrics_df['MAPE'].mean()
                            st.metric("Average MAPE", f"{avg_mape:.1f}%")
                else:
                    st.info("Select models and click Train Models to start")
        else:
            st.warning("Please collect data first")


if __name__ == "__main__":
    main()