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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import time
import json

# Page configuration
st.set_page_config(
    page_title="Solar Power Forecasting - Rural India",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #004E89;
        font-weight: bold;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 10px;
        border-left: 5px solid #28a745;
        background-color: #d4edda;
        margin: 10px 0;
    }
    .info-box {
        padding: 10px;
        border-left: 5px solid #17a2b8;
        background-color: #d1ecf1;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ==================== DATA COLLECTION CLASS ====================
class SolarDataCollector:
    """Collect weather and solar irradiance data using free APIs"""
    
    def __init__(self):
        self.open_meteo_base = "https://archive-api.open-meteo.com/v1/archive"
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def get_weather_data_open_meteo(self, latitude, longitude, start_date, end_date):
        """Get historical weather data from Open-Meteo"""
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,'
                    'precipitation_sum,windspeed_10m_max,shortwave_radiation_sum,'
                    'relative_humidity_2m_mean,cloudcover_mean',
            'timezone': 'Asia/Kolkata'
        }
        
        try:
            response = requests.get(self.open_meteo_base, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' in data:
                df = pd.DataFrame(data['daily'])
                df['date'] = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
                return df
            return None
        except Exception as e:
            st.error(f"Error fetching Open-Meteo data: {e}")
            return None
    
    def get_solar_data_nasa_power(self, latitude, longitude, start_date, end_date):
        """Get solar irradiance data from NASA POWER"""
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        
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
            st.error(f"Error fetching NASA POWER data: {e}")
            return None
    
    def collect_data_for_location(self, location_name, latitude, longitude, 
                                  start_date, end_date):
        """Collect comprehensive data for a location"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"Fetching weather data for {location_name}...")
        progress_bar.progress(25)
        weather_df = self.get_weather_data_open_meteo(latitude, longitude, 
                                                       start_date, end_date)
        time.sleep(1)
        
        status_text.text(f"Fetching solar irradiance data for {location_name}...")
        progress_bar.progress(50)
        solar_df = self.get_solar_data_nasa_power(latitude, longitude, 
                                                   start_date, end_date)
        
        if weather_df is not None and solar_df is not None:
            status_text.text("Merging datasets...")
            progress_bar.progress(75)
            
            merged_df = pd.merge(weather_df, solar_df, on='date', how='inner')
            merged_df['location'] = location_name
            merged_df['latitude'] = latitude
            merged_df['longitude'] = longitude
            
            progress_bar.progress(100)
            status_text.text(f"✓ Successfully collected {len(merged_df)} days of data")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            return merged_df
        else:
            progress_bar.empty()
            status_text.empty()
            return None


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
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Seasonal indicators
        df['season'] = df['month'].apply(lambda x: 
            1 if x in [12, 1, 2] else  # winter
            2 if x in [3, 4, 5] else    # summer
            3 if x in [6, 7, 8, 9] else # monsoon
            4                            # post-monsoon
        )
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    @staticmethod
    def add_solar_features(df):
        """Add solar-specific features"""
        df = df.copy()
        
        # Clearness index
        if 'solar_irradiance_kwh_m2' in df.columns and 'clear_sky_irradiance_kwh_m2' in df.columns:
            df['clearness_index'] = df['solar_irradiance_kwh_m2'] / (df['clear_sky_irradiance_kwh_m2'] + 0.001)
            df['clearness_index'] = df['clearness_index'].clip(0, 1)
        
        # Temperature efficiency factor
        if 'temperature_2m_mean' in df.columns:
            df['temp_efficiency_factor'] = 1 - 0.005 * (df['temperature_2m_mean'] - 25)
        elif 'temperature_c' in df.columns:
            df['temp_efficiency_factor'] = 1 - 0.005 * (df['temperature_c'] - 25)
        
        # Cloud reduction factor
        if 'cloudcover_mean' in df.columns:
            df['cloud_reduction_factor'] = 1 - (df['cloudcover_mean'] / 100) * 0.75
        
        return df
    
    @staticmethod
    def add_lag_features(df, target_col, lags=[1, 7, 30, 365]):
        """Add lagged features"""
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    @staticmethod
    def add_rolling_features(df, target_col, windows=[7, 30, 90]):
        """Add rolling statistics"""
        df = df.copy()
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        return df


# ==================== SOLAR POWER CALCULATOR ====================
def calculate_solar_power(df, panel_capacity_kw=1.0, efficiency=0.15):
    """
    Calculate solar power output from irradiance data
    
    Args:
        df: DataFrame with solar irradiance
        panel_capacity_kw: Installed solar panel capacity in kW
        efficiency: Panel efficiency (default 15%)
    
    Returns:
        DataFrame with solar_power_kwh column
    """
    df = df.copy()
    
    # Solar power = Irradiance * Panel Area * Efficiency
    # Panel capacity = Panel Area * 1 kW/m² * Efficiency
    # So Panel Area = Panel Capacity / Efficiency
    
    if 'solar_irradiance_kwh_m2' in df.columns:
        df['solar_power_kwh'] = (
            df['solar_irradiance_kwh_m2'] * 
            panel_capacity_kw * 
            efficiency
        )
        
        # Apply efficiency factors if available
        if 'temp_efficiency_factor' in df.columns:
            df['solar_power_kwh'] *= df['temp_efficiency_factor']
        
        if 'cloud_reduction_factor' in df.columns:
            df['solar_power_kwh'] *= df['cloud_reduction_factor']
    
    return df


# ==================== MODELS ====================
class RandomForestSolarModel:
    """Random Forest model for solar forecasting"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X, y):
        """Train the model"""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None


class LSTMSolarModel:
    """LSTM model for solar forecasting"""
    
    def __init__(self, lookback=30, features_dim=10):
        self.lookback = lookback
        self.features_dim = features_dim
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_sequences(self, X, y):
        """Prepare sequences for LSTM"""
        X_scaled = self.scaler.fit_transform(X)
        
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - self.lookback):
            X_seq.append(X_scaled[i:i+self.lookback])
            y_seq.append(y.iloc[i+self.lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self):
        """Build LSTM architecture"""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            model = Sequential([
                LSTM(64, activation='relu', return_sequences=True, 
                     input_shape=(self.lookback, self.features_dim)),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.model = model
            return True
        except ImportError:
            st.error("TensorFlow not installed. Install with: pip install tensorflow")
            return False
    
    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        if self.model is None:
            if not self.build_model():
                return None
        
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        return history
    
    def predict(self, X, y_dummy):
        """Make predictions"""
        if self.model is None:
            return None
        
        X_seq, _ = self.prepare_sequences(X, y_dummy)
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()


class ARIMASolarModel:
    """ARIMA model for solar forecasting"""
    
    def __init__(self, order=(5, 1, 2)):
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def train(self, y):
        """Train ARIMA model"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            self.model = ARIMA(y, order=self.order)
            self.fitted_model = self.model.fit()
            return True
        except ImportError:
            st.error("Statsmodels not installed. Install with: pip install statsmodels")
            return False
        except Exception as e:
            st.error(f"ARIMA training error: {e}")
            return False
    
    def predict(self, steps):
        """Make predictions"""
        if self.fitted_model is None:
            return None
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values


class HybridSolarModel:
    """Hybrid model combining RF, LSTM, and ARIMA"""
    
    def __init__(self):
        self.rf_model = RandomForestSolarModel()
        self.lstm_model = None
        self.arima_model = None
        self.weights = {'rf': 0.5, 'lstm': 0.3, 'arima': 0.2}
        
    def train(self, X, y, use_lstm=True, use_arima=True):
        """Train all models"""
        results = {}
        
        # Train Random Forest
        st.info("Training Random Forest model...")
        self.rf_model.train(X, y)
        results['rf'] = True
        
        # Train LSTM if requested
        if use_lstm:
            try:
                st.info("Training LSTM model...")
                self.lstm_model = LSTMSolarModel(lookback=30, features_dim=X.shape[1])
                history = self.lstm_model.train(X, y, epochs=30, batch_size=32)
                results['lstm'] = True if history else False
            except Exception as e:
                st.warning(f"LSTM training failed: {e}")
                results['lstm'] = False
        
        # Train ARIMA if requested
        if use_arima:
            try:
                st.info("Training ARIMA model...")
                self.arima_model = ARIMASolarModel(order=(5, 1, 2))
                success = self.arima_model.train(y)
                results['arima'] = success
            except Exception as e:
                st.warning(f"ARIMA training failed: {e}")
                results['arima'] = False
        
        return results
    
    def predict(self, X, y_dummy=None, steps=None):
        """Make hybrid predictions"""
        predictions = []
        active_models = []
        
        # Random Forest prediction
        rf_pred = self.rf_model.predict(X)
        predictions.append(rf_pred * self.weights['rf'])
        active_models.append('rf')
        
        # LSTM prediction
        if self.lstm_model and self.lstm_model.model:
            try:
                lstm_pred = self.lstm_model.predict(X, y_dummy)
                if lstm_pred is not None and len(lstm_pred) == len(rf_pred):
                    # Pad LSTM predictions to match RF length
                    lstm_full = np.zeros(len(rf_pred))
                    lstm_full[-len(lstm_pred):] = lstm_pred
                    predictions.append(lstm_full * self.weights['lstm'])
                    active_models.append('lstm')
            except:
                pass
        
        # ARIMA prediction
        if self.arima_model and self.arima_model.fitted_model and steps:
            try:
                arima_pred = self.arima_model.predict(steps)
                if arima_pred is not None and len(arima_pred) == len(rf_pred):
                    predictions.append(arima_pred * self.weights['arima'])
                    active_models.append('arima')
            except:
                pass
        
        # Combine predictions
        hybrid_pred = np.sum(predictions, axis=0)
        
        # Normalize if not all models were used
        total_weight = sum(self.weights[m] for m in active_models)
        if total_weight < 1.0 and total_weight > 0:
            hybrid_pred = hybrid_pred / total_weight
        
        return hybrid_pred


# ==================== EVALUATION METRICS ====================
def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }


# ==================== VISUALIZATION FUNCTIONS ====================
def plot_time_series(df, location_name):
    """Plot time series data"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Solar Irradiance', 'Temperature', 'Cloud Cover'),
        vertical_spacing=0.1
    )
    
    # Solar Irradiance
    if 'solar_irradiance_kwh_m2' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['solar_irradiance_kwh_m2'],
                      name='Solar Irradiance', line=dict(color='orange')),
            row=1, col=1
        )
    
    # Temperature
    if 'temperature_2m_mean' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['temperature_2m_mean'],
                      name='Temperature', line=dict(color='red')),
            row=2, col=1
        )
    elif 'temperature_c' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['temperature_c'],
                      name='Temperature', line=dict(color='red')),
            row=2, col=1
        )
    
    # Cloud Cover
    if 'cloudcover_mean' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['cloudcover_mean'],
                      name='Cloud Cover', line=dict(color='gray')),
            row=3, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="kWh/m²", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    
    fig.update_layout(height=800, title_text=f"Weather Data - {location_name}",
                     showlegend=True)
    
    return fig


def plot_seasonal_patterns(df):
    """Plot seasonal patterns"""
    df_monthly = df.copy()
    df_monthly['year_month'] = df_monthly['date'].dt.to_period('M')
    
    monthly_avg = df_monthly.groupby('year_month').agg({
        'solar_irradiance_kwh_m2': 'mean',
        'temperature_2m_mean': 'mean' if 'temperature_2m_mean' in df.columns else 
                               ('temperature_c' if 'temperature_c' in df.columns else None)
    }).reset_index()
    
    monthly_avg['year_month'] = monthly_avg['year_month'].astype(str)
    
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=('Monthly Solar Irradiance', 'Monthly Temperature'))
    
    fig.add_trace(
        go.Bar(x=monthly_avg['year_month'], y=monthly_avg['solar_irradiance_kwh_m2'],
              name='Solar Irradiance', marker_color='orange'),
        row=1, col=1
    )
    
    temp_col = 'temperature_2m_mean' if 'temperature_2m_mean' in monthly_avg.columns else 'temperature_c'
    if temp_col in monthly_avg.columns:
        fig.add_trace(
            go.Bar(x=monthly_avg['year_month'], y=monthly_avg[temp_col],
                  name='Temperature', marker_color='red'),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="Month", tickangle=45)
    fig.update_yaxes(title_text="kWh/m²", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def plot_predictions(y_true, y_pred, dates, title="Model Predictions"):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        mode='lines',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Solar Power (kWh)",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def plot_feature_importance(importance_df):
    """Plot feature importance"""
    top_features = importance_df.head(15)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker=dict(color=top_features['importance'],
                   colorscale='Viridis')
    ))
    
    fig.update_layout(
        title="Top 15 Feature Importance (Random Forest)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


# ==================== MAIN STREAMLIT APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">☀️ Solar Power Forecasting for Rural India</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Project Overview:</b> Hybrid predictive algorithms combining Machine Learning (Random Forest, LSTM) 
    and Statistical methods (ARIMA) to forecast long-term solar power output using publicly available 
    weather patterns and solar irradiance data.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Predefined rural locations in India
    RURAL_LOCATIONS = {
        'Jodhpur, Rajasthan': (26.2389, 73.0243),
        'Jaisalmer, Rajasthan': (26.9157, 70.9083),
        'Anantapur, Andhra Pradesh': (14.6819, 77.6006),
        'Bikaner, Rajasthan': (28.0229, 73.3119),
        'Kurnool, Andhra Pradesh': (15.8281, 78.0373),
        'Nashik, Maharashtra': (19.9975, 73.7898),
        'Solapur, Maharashtra': (17.6599, 75.9064),
        'Custom Location': None
    }
    
    location_choice = st.sidebar.selectbox(
        "Select Location",
        list(RURAL_LOCATIONS.keys())
    )
    
    if location_choice == 'Custom Location':
        location_name = st.sidebar.text_input("Location Name", "My Location")
        latitude = st.sidebar.number_input("Latitude", value=26.2389, format="%.4f")
        longitude = st.sidebar.number_input("Longitude", value=73.0243, format="%.4f")
    else:
        location_name = location_choice
        latitude, longitude = RURAL_LOCATIONS[location_choice]
        st.sidebar.info(f"📍 Coordinates: {latitude}°N, {longitude}°E")
    
    # Date range
    st.sidebar.subheader("📅 Data Collection Period")
    default_start = datetime.now() - timedelta(days=3*365)
    default_end = datetime.now() - timedelta(days=1)
    
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", default_end)
    
    # Solar panel configuration
    st.sidebar.subheader("⚡ Solar Panel Configuration")
    panel_capacity = st.sidebar.number_input("Panel Capacity (kW)", value=1.0, step=0.1)
    panel_efficiency = st.sidebar.slider("Panel Efficiency (%)", 10, 25, 15) / 100
    
    # Model configuration
    st.sidebar.subheader("🤖 Model Configuration")
    use_random_forest = st.sidebar.checkbox("Random Forest", value=True)
    use_lstm = st.sidebar.checkbox("LSTM (Deep Learning)", value=False)
    use_arima = st.sidebar.checkbox("ARIMA (Statistical)", value=False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Collection", 
        "🔍 Data Analysis", 
        "🤖 Model Training", 
        "📈 Predictions", 
        "📚 Documentation"
    ])
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # ==================== TAB 1: DATA COLLECTION ====================
    with tab1:
        st.markdown('<p class="sub-header">Data Collection</p>', unsafe_allow_html=True)
        
        st.write(f"""
        **Location:** {location_name}  
        **Coordinates:** {latitude}°N, {longitude}°E  
        **Period:** {start_date} to {end_date}  
        **Panel Capacity:** {panel_capacity} kW  
        **Panel Efficiency:** {panel_efficiency*100}%
        """)
        
        if st.button("🚀 Collect Data", type="primary", width='stretch'):
            collector = SolarDataCollector()
            
            df = collector.collect_data_for_location(
                location_name, latitude, longitude,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if df is not None and len(df) > 0:
                st.success(f"✅ Successfully collected {len(df)} days of data!")
                
                # Feature engineering
                st.info("Creating features...")
                fe = SolarFeatureEngineering()
                df = fe.add_temporal_features(df)
                df = fe.add_solar_features(df)
                df = calculate_solar_power(df, panel_capacity, panel_efficiency)
                
                # Add lag and rolling features for solar power
                if 'solar_power_kwh' in df.columns:
                    df = fe.add_lag_features(df, 'solar_power_kwh', lags=[1, 7, 30])
                    df = fe.add_rolling_features(df, 'solar_power_kwh', windows=[7, 30, 90])
                
                # Drop rows with NaN values
                df = df.dropna()
                
                st.session_state.data = df
                
                # Display data summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Days", len(df))
                with col2:
                    if 'solar_power_kwh' in df.columns:
                        st.metric("Avg Solar Power", f"{df['solar_power_kwh'].mean():.2f} kWh")
                with col3:
                    if 'solar_irradiance_kwh_m2' in df.columns:
                        st.metric("Avg Irradiance", f"{df['solar_irradiance_kwh_m2'].mean():.2f} kWh/m²")
                with col4:
                    temp_col = 'temperature_2m_mean' if 'temperature_2m_mean' in df.columns else 'temperature_c'
                    if temp_col in df.columns:
                        st.metric("Avg Temperature", f"{df[temp_col].mean():.1f} °C")
                
                # Display sample data
                st.subheader("Sample Data")
                st.dataframe(df.head(10), width='stretch')
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"solar_data_{location_name}_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Failed to collect data. Please check your inputs and try again.")
    
    # ==================== TAB 2: DATA ANALYSIS ====================
    with tab2:
        st.markdown('<p class="sub-header">Data Analysis & Visualization</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Time series plot
            st.subheader("📊 Time Series Data")
            fig_ts = plot_time_series(df, location_name)
            st.plotly_chart(fig_ts, width='stretch')
            
            # Seasonal patterns
            st.subheader("📅 Seasonal Patterns")
            fig_seasonal = plot_seasonal_patterns(df)
            st.plotly_chart(fig_seasonal, width='stretch')
            
            # Statistical summary
            st.subheader("📈 Statistical Summary")
            
            cols_to_describe = ['solar_irradiance_kwh_m2', 'solar_power_kwh']
            temp_col = 'temperature_2m_mean' if 'temperature_2m_mean' in df.columns else 'temperature_c'
            if temp_col in df.columns:
                cols_to_describe.append(temp_col)
            
            available_cols = [col for col in cols_to_describe if col in df.columns]
            
            if available_cols:
                st.dataframe(df[available_cols].describe(), width='stretch')
            
            # Correlation matrix
            st.subheader("🔗 Correlation Matrix")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude lag and rolling features for cleaner visualization
            core_cols = [col for col in numeric_cols if 'lag' not in col and 'rolling' not in col][:10]
            
            if len(core_cols) > 1:
                corr_matrix = df[core_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix'
                )
                st.plotly_chart(fig_corr, width='stretch')
        else:
            st.warning("⚠️ Please collect data first in the 'Data Collection' tab.")
    
    # ==================== TAB 3: MODEL TRAINING ====================
    with tab3:
        st.markdown('<p class="sub-header">Model Training</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            if 'solar_power_kwh' not in df.columns:
                st.error("Solar power column not found. Please recollect data.")
            else:
                # Prepare features and target
                st.subheader("🔧 Feature Preparation")
                
                exclude_cols = ['date', 'location', 'solar_power_kwh']
                feature_cols = [col for col in df.columns if col not in exclude_cols]
                
                X = df[feature_cols]
                y = df['solar_power_kwh']
                
                st.write(f"**Features:** {len(feature_cols)} features")
                st.write(f"**Target:** solar_power_kwh")
                st.write(f"**Samples:** {len(X)} days")
                
                # Train-test split
                test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
                
                # Time-based split (more appropriate for time series)
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                dates_train, dates_test = df['date'][:split_idx], df['date'][split_idx:]
                
                st.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
                
                # Train button
                if st.button("🎯 Train Models", type="primary", width='stretch'):
                    with st.spinner("Training models... This may take a few minutes."):
                        # Create hybrid model
                        hybrid_model = HybridSolarModel()
                        
                        # Train
                        results = hybrid_model.train(
                            X_train, y_train,
                            use_lstm=use_lstm,
                            use_arima=use_arima
                        )
                        
                        st.session_state.model = hybrid_model
                        st.session_state.trained = True
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.dates_train = dates_train
                        st.session_state.dates_test = dates_test
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Show which models were trained
                        trained_models = [k for k, v in results.items() if v]
                        st.write(f"**Trained models:** {', '.join(trained_models)}")
                        
                        # Feature importance (from Random Forest)
                        if use_random_forest:
                            st.subheader("📊 Feature Importance")
                            importance_df = hybrid_model.rf_model.get_feature_importance()
                            if importance_df is not None:
                                fig_importance = plot_feature_importance(importance_df)
                                st.plotly_chart(fig_importance, width='stretch')
        else:
            st.warning("⚠️ Please collect data first in the 'Data Collection' tab.")
    
    # ==================== TAB 4: PREDICTIONS ====================
    with tab4:
        st.markdown('<p class="sub-header">Model Predictions & Evaluation</p>', unsafe_allow_html=True)
        
        if st.session_state.trained and st.session_state.model is not None:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            dates_test = st.session_state.dates_test
            
            # Make predictions
            with st.spinner("Generating predictions..."):
                y_pred = model.predict(X_test, y_test, steps=len(X_test))
            
            # Calculate metrics
            metrics = calculate_metrics(y_test.values, y_pred)
            
            # Display metrics
            st.subheader("📊 Model Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
            with col2:
                st.metric("MAE", f"{metrics['MAE']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics['R²']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with col5:
                st.metric("MSE", f"{metrics['MSE']:.4f}")
            
            # Prediction plot
            st.subheader("📈 Actual vs Predicted Solar Power")
            fig_pred = plot_predictions(y_test.values, y_pred, dates_test.values,
                                       "Hybrid Model: Actual vs Predicted")
            st.plotly_chart(fig_pred, width='stretch')
            
            # Residuals plot
            st.subheader("📉 Prediction Residuals")
            residuals = y_test.values - y_pred
            
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=dates_test.values, y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='purple', size=5)
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residuals.update_layout(
                title="Prediction Residuals (Actual - Predicted)",
                xaxis_title="Date",
                yaxis_title="Residual (kWh)",
                height=400
            )
            st.plotly_chart(fig_residuals, width='stretch')
            
            # Monthly aggregation
            st.subheader("📅 Monthly Performance")
            results_df = pd.DataFrame({
                'date': dates_test.values,
                'actual': y_test.values,
                'predicted': y_pred
            })
            results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')
            
            monthly_perf = results_df.groupby('month').agg({
                'actual': 'sum',
                'predicted': 'sum'
            }).reset_index()
            monthly_perf['month'] = monthly_perf['month'].astype(str)
            
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_perf['month'],
                y=monthly_perf['actual'],
                name='Actual',
                marker_color='blue'
            ))
            fig_monthly.add_trace(go.Bar(
                x=monthly_perf['month'],
                y=monthly_perf['predicted'],
                name='Predicted',
                marker_color='red'
            ))
            fig_monthly.update_layout(
                title="Monthly Solar Power Generation",
                xaxis_title="Month",
                yaxis_title="Total Solar Power (kWh)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_monthly, width='stretch')
            
            # Download predictions
            results_df_download = results_df[['date', 'actual', 'predicted']].copy()
            results_df_download['residual'] = results_df_download['actual'] - results_df_download['predicted']
            
            csv_pred = results_df_download.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_pred,
                file_name=f"solar_predictions_{location_name}.csv",
                mime="text/csv"
            )
            
        elif st.session_state.data is not None:
            st.warning("⚠️ Please train the models first in the 'Model Training' tab.")
        else:
            st.warning("⚠️ Please collect data first in the 'Data Collection' tab.")
    
    # ==================== TAB 5: DOCUMENTATION ====================
    with tab5:
        st.markdown('<p class="sub-header">Documentation & Guide</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ## 🎯 Project Overview
        
        This application implements a **hybrid predictive algorithm** to forecast long-term solar power 
        output in rural regions of India. It combines:
        
        - **Machine Learning**: Random Forest, LSTM (Deep Learning)
        - **Statistical Methods**: ARIMA for time series forecasting
        - **Data Sources**: Free APIs (Open-Meteo, NASA POWER)
        
        ---
        
        ## 📚 How to Use
        
        ### 1. Data Collection
        - Select a location from predefined rural areas or enter custom coordinates
        - Choose date range (recommended: 2-3 years for better model training)
        - Configure solar panel capacity and efficiency
        - Click "Collect Data" to fetch weather and solar irradiance data
        
        ### 2. Data Analysis
        - Explore time series patterns
        - Analyze seasonal trends
        - View statistical summaries and correlations
        
        ### 3. Model Training
        - Choose which models to train (RF, LSTM, ARIMA)
        - Adjust train-test split ratio
        - Train models and view feature importance
        
        ### 4. Predictions & Evaluation
        - View model predictions vs actual values
        - Analyze performance metrics (RMSE, MAE, R², MAPE)
        - Examine residuals and monthly aggregations
        - Download predictions for further analysis
        
        ---
        
        ## 🔬 Methodology
        
        ### Data Sources
        1. **Open-Meteo API** (Free): Temperature, humidity, wind, precipitation, cloud cover
        2. **NASA POWER API** (Free): Solar irradiance, clear-sky irradiance
        
        ### Feature Engineering
        - **Temporal Features**: Month, quarter, season, day of year (with cyclical encoding)
        - **Solar Features**: Clearness index, temperature efficiency factor, cloud reduction
        - **Lag Features**: Past values at 1, 7, 30 days
        - **Rolling Features**: Moving averages and standard deviations
        
        ### Models
        
        #### 1. Random Forest
        - Captures non-linear relationships
        - Handles multivariate weather patterns
        - Provides feature importance
        
        #### 2. LSTM (Long Short-Term Memory)
        - Deep learning for sequential patterns
        - Captures long-term dependencies
        - Learns temporal dynamics
        
        #### 3. ARIMA (AutoRegressive Integrated Moving Average)
        - Statistical time series method
        - Models linear trends and seasonality
        - Good for stable, predictable patterns
        
        #### 4. Hybrid Ensemble
        - Combines all models with weighted averaging
        - Leverages strengths of each approach
        - More robust predictions
        
        ---
        
        ## 📊 Performance Metrics
        
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **R²**: Coefficient of determination (closer to 1 is better)
        - **MAPE**: Mean Absolute Percentage Error (lower is better)
        
        ---
        
        ## 🌍 Rural India Focus
        
        This tool is specifically designed for:
        - Decentralized solar energy systems
        - Off-grid rural communities
        - Solar mini-grids and microgrids
        - Policy planning for renewable energy deployment
        - Sustainable development initiatives
        
        ---
        
        ## 💡 Tips for Best Results
        
        1. **Data Period**: Use at least 2-3 years of historical data
        2. **Location Selection**: Choose locations with good solar potential
        3. **Model Selection**: Use all three models for best hybrid results
        4. **Validation**: Always check residuals and monthly aggregations
        5. **Seasonality**: Account for monsoon and seasonal variations
        
        ---
        
        ## 🛠️ Technical Requirements
        
        ```bash
        # Install required packages
        pip install streamlit pandas numpy plotly scikit-learn requests
        
        # Optional (for LSTM)
        pip install tensorflow
        
        # Optional (for ARIMA)
        pip install statsmodels
        ```
        
        ---
        
        ## 📖 References
        
        - **Open-Meteo**: https://open-meteo.com/
        - **NASA POWER**: https://power.larc.nasa.gov/
        - **Scikit-learn**: https://scikit-learn.org/
        - **TensorFlow**: https://www.tensorflow.org/
        - **Statsmodels**: https://www.statsmodels.org/
        
        ---
        
        ## 🤝 Support & Feedback
        
        For questions, suggestions, or issues:
        - Check API documentation for data limitations
        - Ensure stable internet connection for data collection
        - Adjust model parameters based on your specific use case
        
        ---
        
        ## 📄 License & Citation
        
        If you use this tool for research or publications, please cite appropriately and acknowledge 
        the data sources (Open-Meteo and NASA POWER).
        """)


# Run the app
if __name__ == "__main__":
    main()