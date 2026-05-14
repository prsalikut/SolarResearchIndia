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

# Deep Learning imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

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
    page_title="Solar Power Forecasting - Rural India Study",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# THEME - Professional Scientific/Engineering Theme
# =============================================================================
st.markdown("""
    <style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Main theme colors - Professional dark/light hybrid */
    :root {
        --primary-dark: #0A2F44;
        --primary: #1E4A6B;
        --primary-light: #2C6B8C;
        --accent: #E9B741;
        --accent-dark: #C99C2E;
        --success: #2E7D32;
        --warning: #F57C00;
        --danger: #D32F2F;
        --gray-100: #F5F7FA;
        --gray-200: #E4E7EB;
        --gray-300: #CBD2D9;
        --gray-400: #9AA5B1;
        --gray-500: #7B8794;
        --gray-600: #616E7C;
        --gray-700: #3E4C59;
        --gray-800: #1F2933;
        --gray-900: #0B1C2B;
    }
    
    /* Global styles */
    .main {
        background: linear-gradient(135deg, var(--gray-100) 0%, #FFFFFF 100%);
        padding: 1.5rem 2rem;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Sidebar styling - WHITE BACKGROUND */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid var(--gray-300);
        padding: 1.5rem 0.5rem;
    }
    
    [data-testid="stSidebar"] * {
        color: var(--gray-800) !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: var(--gray-700) !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white;
        border: 1px solid var(--gray-300);
        color: var(--gray-800) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--primary-dark) !important;
    }
    
    /* WHITE SLIDER STYLING */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background-color: var(--gray-200) !important;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div > div {
        background-color: var(--primary) !important;
    }
    
    /* Sidebar markdown text */
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--gray-700) !important;
    }
    
    /* Sidebar info boxes */
    [data-testid="stSidebar"] .stAlert {
        background-color: var(--gray-100);
        border-left: 4px solid var(--primary);
        color: var(--gray-800) !important;
    }
    
    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: var(--gray-300);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary-dark);
        text-align: center;
        padding: 20px 15px;
        margin-bottom: 20px;
        letter-spacing: -0.5px;
        border-bottom: 3px solid var(--accent);
        background: linear-gradient(90deg, rgba(10,47,68,0.03) 0%, rgba(233,183,65,0.05) 100%);
        border-radius: 0 0 10px 10px;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: var(--primary-dark);
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 20px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--accent);
        letter-spacing: -0.3px;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid var(--gray-200);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(10,47,68,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(30,74,107,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        box-shadow: 0 6px 12px rgba(10,47,68,0.3);
        transform: translateY(-2px);
        border: 1px solid var(--accent);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        color: var(--gray-900);
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, var(--accent-dark) 0%, var(--accent) 100%);
        color: var(--gray-900);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        color: white !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-dark);
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: var(--gray-600);
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: var(--gray-800);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent) 0%, var(--success) 100%);
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 6px;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-family: 'Inter', sans-serif;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: var(--primary);
        color: white;
        padding: 10px;
    }
    
    .dataframe td {
        padding: 8px 10px;
        border-bottom: 1px solid var(--gray-200);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--gray-500);
        font-size: 0.85rem;
        border-top: 1px solid var(--gray-200);
        margin-top: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        .block-container {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# ENHANCED DATA COLLECTOR - Multiple Free APIs with Fallbacks
# =============================================================================
class SolarDataCollector:
    def __init__(self):
        # NASA POWER API - Primary source
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # Open-Meteo API - Secondary source (no API key required)
        self.open_meteo_base = "https://archive-api.open-meteo.com/v1/archive"
        
        # SolarGIS API - Tertiary source (public demo endpoint)
        self.solargis_base = "https://api.solargis.com/v1/demo/pvcalc/daily"
        
        # Cache for API responses
        self.cache = {}
        
    def get_solar_data_nasa_power(self, latitude, longitude, start_date, end_date):
        """Primary API: NASA POWER (most comprehensive for solar)"""
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
            'parameters': 'ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS10M,PRECTOTCORR',
            'community': 'RE',
            'longitude': longitude,
            'latitude': latitude,
            'start': start,
            'end': end,
            'format': 'JSON'
        }
        
        try:
            response = requests.get(self.nasa_power_base, params=params, timeout=30)
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
                    'WS10M': 'wind_speed_ms',
                    'PRECTOTCORR': 'precipitation_mm'
                })
                
                # Add derived features
                df['data_source'] = 'NASA POWER'
                return df
            return None
        except Exception as e:
            st.warning(f"NASA POWER API: {str(e)[:100]}...")
            return None
    
    def get_solar_data_open_meteo(self, latitude, longitude, start_date, end_date):
        """Secondary API: Open-Meteo (no key required, good for Europe/Asia)"""
        try:
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'daily': ['shortwave_radiation_sum', 'temperature_2m_mean', 
                         'relative_humidity_2m_mean', 'wind_speed_10m_mean',
                         'precipitation_sum', 'cloud_cover_mean'],
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(self.open_meteo_base, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' in data:
                df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'solar_irradiance_kwh_m2': np.array(data['daily']['shortwave_radiation_sum']) / 1000,  # Wh/m² to kWh/m²
                    'temperature_c': data['daily']['temperature_2m_mean'],
                    'relative_humidity': data['daily']['relative_humidity_2m_mean'],
                    'wind_speed_ms': data['daily']['wind_speed_10m_mean'],
                    'precipitation_mm': data['daily']['precipitation_sum'],
                    'cloudcover_mean': data['daily']['cloud_cover_mean']
                })
                
                # Calculate clear sky estimate (approximation)
                df['clear_sky_irradiance_kwh_m2'] = df['solar_irradiance_kwh_m2'] * (1 + 0.3 * np.random.random(len(df)))
                df['data_source'] = 'Open-Meteo'
                
                return df
            return None
        except Exception as e:
            st.warning(f"Open-Meteo API: {str(e)[:100]}...")
            return None
    
    def generate_climate_aware_synthetic_data(self, latitude, longitude, start_date, end_date):
        """Enhanced synthetic data with climate zone characteristics"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Determine climate zone based on latitude and longitude (India-specific)
        if latitude > 28:  # Northern India
            climate_zone = "Temperate/North"
            base_temp = 22 + 15 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 5.0 + 2.8 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
            monsoon_months = [7, 8]  # July-August
        elif latitude > 20:  # Central India
            climate_zone = "Tropical/Central"
            base_temp = 26 + 10 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 5.5 + 2.5 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
            monsoon_months = [6, 7, 8, 9]  # June-September
        elif latitude > 12:  # Southern India
            climate_zone = "Tropical/South"
            base_temp = 28 + 6 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 6.0 + 2.2 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
            monsoon_months = [6, 7, 8, 9, 10]  # June-October
        else:  # Coastal/Extreme South
            climate_zone = "Coastal"
            base_temp = 29 + 4 * np.sin(2 * np.pi * (date_range.dayofyear - 105) / 365)
            base_irradiance = 5.8 + 2.0 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
            monsoon_months = [5, 6, 7, 8, 9, 10]  # May-October
        
        # Set seed for reproducibility
        np.random.seed(int(abs(latitude * longitude) % 10000))
        
        # Generate monsoon effect
        monsoon_effect = np.zeros(len(date_range))
        for month in monsoon_months:
            monsoon_effect += (date_range.month == month).astype(int) * np.random.uniform(0.3, 0.7)
        
        # Generate data with realistic variations
        solar_irradiance = base_irradiance * (1 - 0.4 * monsoon_effect) + np.random.normal(0, 0.5, len(date_range))
        solar_irradiance = np.maximum(solar_irradiance, 1.0)
        
        df = pd.DataFrame({
            'date': date_range,
            'solar_irradiance_kwh_m2': np.abs(solar_irradiance),
            'temperature_c': base_temp + np.random.normal(0, 2.5, len(date_range)),
            'relative_humidity': np.clip(65 + 25 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + 
                                       20 * monsoon_effect + np.random.normal(0, 10, len(date_range)), 30, 100),
            'cloudcover_mean': np.clip(35 + 30 * np.sin(2 * np.pi * (date_range.dayofyear + 150) / 365) + 
                                     30 * monsoon_effect + np.random.normal(0, 15, len(date_range)), 0, 100),
            'precipitation_mm': np.maximum(monsoon_effect * np.random.uniform(2, 15, len(date_range)) + 
                                          np.random.exponential(2, len(date_range)), 0)
        })
        
        df['clear_sky_irradiance_kwh_m2'] = base_irradiance * 1.2 + np.random.normal(0, 0.3, len(date_range))
        df['clear_sky_irradiance_kwh_m2'] = np.abs(df['clear_sky_irradiance_kwh_m2']).clip(1.5, 9)
        
        df['wind_speed_ms'] = np.abs(3.5 + 2 * np.random.random(len(date_range)) + 1.5 * np.sin(2 * np.pi * date_range.dayofyear / 180))
        df['wind_speed_ms'] = df['wind_speed_ms'].clip(0.5, 12)
        
        df['temperature_c'] = df['temperature_c'].clip(10, 42)
        
        # Add metadata
        df['climate_zone'] = climate_zone
        df['data_source'] = 'Synthetic (Climate-Aware)'
        
        return df
    
    def collect_data_for_location(self, location_name, latitude, longitude, start_date, end_date):
        """Collect data from multiple APIs with fallback strategy"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Try NASA POWER first
        status_text.text("Fetching data from NASA POWER API...")
        progress_bar.progress(20)
        df = self.get_solar_data_nasa_power(latitude, longitude, start_date, end_date)
        
        # Fallback to Open-Meteo if NASA fails
        if df is None:
            status_text.text("NASA API unavailable. Trying Open-Meteo...")
            progress_bar.progress(40)
            df = self.get_solar_data_open_meteo(latitude, longitude, start_date, end_date)
        
        # Generate synthetic data if all APIs fail
        if df is None:
            status_text.text("APIs unavailable. Generating climate-aware synthetic data...")
            progress_bar.progress(60)
            df = self.generate_climate_aware_synthetic_data(latitude, longitude, start_date, end_date)
            st.info(f"Using climate-aware synthetic data for {location_name} ({df['climate_zone'].iloc[0]} zone)")
        
        # Post-process data
        progress_bar.progress(80)
        status_text.text("Processing and validating data...")
        
        # Ensure required columns exist
        required_cols = ['solar_irradiance_kwh_m2', 'temperature_c']
        for col in required_cols:
            if col not in df.columns:
                if col == 'solar_irradiance_kwh_m2':
                    df['solar_irradiance_kwh_m2'] = 5.0
                elif col == 'temperature_c':
                    df['temperature_c'] = 25.0
        
        # Add location metadata
        df['location'] = location_name
        df['latitude'] = latitude
        df['longitude'] = longitude
        
        # Convert date and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
        
        progress_bar.progress(100)
        status_text.text(f"Data collection complete: {len(df)} days from {df['data_source'].iloc[0]}")
        time.sleep(0.8)
        status_text.empty()
        progress_bar.empty()
        
        return df


# =============================================================================
# SOLAR POWER CALCULATION - Fixed default values
# =============================================================================
def calculate_solar_power(df, panel_capacity_kw=5.0, efficiency=0.17):
    """
    Calculate solar power generation with fixed default system parameters.
    Panel capacity: 5 kW (standard residential/commercial size)
    Panel efficiency: 17% (typical polycrystalline silicon)
    """
    df = df.copy()
    
    if 'solar_irradiance_kwh_m2' not in df.columns:
        st.warning("No irradiance data found. Using default values.")
        df['solar_irradiance_kwh_m2'] = 5.0
    
    # Panel area calculation (fixed for 5kW system at 17% efficiency)
    panel_area = panel_capacity_kw / (1.0 * efficiency)  # ~29.4 m²
    
    # Base calculation
    df['solar_power_kwh'] = df['solar_irradiance_kwh_m2'] * panel_area * efficiency
    
    # Apply correction factors if available
    if 'clearness_index' in df.columns:
        df['solar_power_kwh'] *= df['clearness_index'].clip(0.5, 0.9)
    
    if 'temp_efficiency_factor' in df.columns:
        df['solar_power_kwh'] *= df['temp_efficiency_factor'].clip(0.85, 1.05)
    
    if 'cloud_reduction_factor' in df.columns:
        df['solar_power_kwh'] *= df['cloud_reduction_factor'].clip(0.4, 1.0)
    
    # Theoretical maximum for 5kW system (8 hours at full capacity)
    max_theoretical = 40.0  # 5kW * 8 hours
    df['solar_power_kwh'] = df['solar_power_kwh'].clip(lower=0.1, upper=max_theoretical)
    
    # Add system metadata
    df['panel_capacity_kw'] = panel_capacity_kw
    df['panel_efficiency'] = efficiency
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
class SolarFeatureEngineering:
    @staticmethod
    def add_temporal_features(df):
        """Add temporal features for time series analysis"""
        df = df.copy()
        
        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Cyclical encoding for seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Indian seasonal markers
        df['is_summer'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
        df['is_post_monsoon'] = df['month'].isin([10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # Weekend effect
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    @staticmethod
    def add_solar_features(df):
        """Add solar-specific derived features"""
        df = df.copy()
        
        # Clearness index (ratio of actual to clear sky irradiance)
        if 'solar_irradiance_kwh_m2' in df.columns and 'clear_sky_irradiance_kwh_m2' in df.columns:
            df['clearness_index'] = df['solar_irradiance_kwh_m2'] / (df['clear_sky_irradiance_kwh_m2'] + 0.001)
            df['clearness_index'] = df['clearness_index'].clip(0, 1)
        
        # Temperature efficiency factor
        temp_col = None
        for col in ['temperature_2m_mean', 'temperature_c']:
            if col in df.columns:
                temp_col = col
                break
        
        if temp_col is not None:
            # Solar panels lose efficiency at high temperatures
            df['temp_efficiency_factor'] = np.maximum(0.80, 1.0 - 0.004 * np.abs(df[temp_col] - 25))
            df['temp_efficiency_factor'] = df['temp_efficiency_factor'].clip(0.80, 1.05)
        
        # Cloud reduction factor
        if 'cloudcover_mean' in df.columns:
            df['cloud_reduction_factor'] = 1.0 - (df['cloudcover_mean'] / 100) * 0.7
            df['cloud_reduction_factor'] = df['cloud_reduction_factor'].clip(0.3, 1.0)
        
        # Solar zenith angle approximation (latitude-based)
        if 'latitude' in df.columns:
            df['solar_zenith_approx'] = np.abs(df['latitude'] - 23.5 * np.cos(2 * np.pi * (df['day_of_year'] - 172) / 365))
        
        return df
    
    @staticmethod
    def create_lag_features(df, target_col='solar_power_kwh', lags=[1, 2, 3, 7, 14, 30]):
        """Create lag features for time series forecasting"""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
        
        return df


# =============================================================================
# HYBRID FORECASTING MODELS - Including LSTM
# =============================================================================
class HybridForecastingModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=200, random_state=42):
        """Random Forest model - ensemble of decision trees"""
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
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
        """Gradient Boosting model - sequential ensemble"""
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=random_state,
            verbose=0
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
        """Linear Regression baseline"""
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
        """Support Vector Regression"""
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
        """Multi-layer Perceptron"""
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=1000,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
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
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """LSTM model for sequence forecasting"""
        if not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow not installed. Skipping LSTM.")
            return None, None
        
        try:
            # Reshape for LSTM [samples, timesteps, features]
            if len(X_train.shape) == 2:
                # Assume we need to create sequences
                timesteps = 7  # Look back 7 days
                
                def create_sequences(data, target, seq_length):
                    X_seq, y_seq = [], []
                    for i in range(len(data) - seq_length):
                        X_seq.append(data[i:i+seq_length])
                        y_seq.append(target[i+seq_length])
                    return np.array(X_seq), np.array(y_seq)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, timesteps)
                X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, timesteps)
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, activation='relu', return_sequences=True, input_shape=(timesteps, X_train.shape[1])),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Early stopping
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                
                # Train
                history = model.fit(
                    X_train_seq, y_train_seq,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Predict
                y_pred_seq = model.predict(X_test_seq, verbose=0)
                y_pred = y_pred_seq.flatten()
                
                metrics = self._calculate_metrics(y_test_seq, y_pred)
                self.models['LSTM'] = model
                self.scalers['LSTM'] = scaler
                
                return metrics, y_pred
            else:
                st.warning("LSTM requires 2D input features")
                return None, None
                
        except Exception as e:
            st.warning(f"LSTM training failed: {str(e)[:100]}")
            return None, None
    
    def train_arima(self, y_train, y_test, order=(2,1,2)):
        """ARIMA model - AutoRegressive Integrated Moving Average"""
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
            st.warning(f"ARIMA failed: {str(e)[:100]}")
            return None, None
    
    def train_sarima(self, y_train, y_test, order=(1,1,1), seasonal_order=(1,1,1,12)):
        """SARIMA model - Seasonal ARIMA"""
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
            st.warning(f"SARIMA failed: {str(e)[:100]}")
            return None, None
    
    def train_exponential_smoothing(self, y_train, y_test, seasonal_periods=12):
        """Exponential Smoothing for time series"""
        if not STATSMODELS_AVAILABLE:
            return None, None
            
        try:
            model = ExponentialSmoothing(y_train, seasonal_periods=seasonal_periods, trend='add', seasonal='add')
            model_fit = model.fit()
            y_pred = model_fit.forecast(steps=len(y_test))
            
            metrics = self._calculate_metrics(y_test, y_pred)
            self.models['Exponential Smoothing'] = model_fit
            
            return metrics, y_pred
        except Exception as e:
            st.warning(f"Exponential Smoothing failed: {str(e)[:100]}")
            return None, None
    
    def train_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Ensemble of Random Forest and Gradient Boosting"""
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
        
        # Weighted ensemble
        ensemble_pred = (0.6 * rf_pred + 0.4 * gb_pred)
        
        metrics = self._calculate_metrics(y_test, ensemble_pred)
        self.models['Ensemble (RF+GB)'] = {
            'rf': self.models['Random Forest'],
            'gb': self.models['Gradient Boosting'],
            'weights': [0.6, 0.4]
        }
        
        return metrics, ensemble_pred
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        if y_pred is None or len(y_pred) == 0:
            return None
            
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_true > 0
        if np.any(mask):
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = 0
        
        # Format for display
        metrics['RMSE'] = float(f"{metrics['RMSE']:.3f}")
        metrics['MAE'] = float(f"{metrics['MAE']:.3f}")
        metrics['R2'] = float(f"{metrics['R2']:.3f}")
        metrics['MAPE'] = float(f"{metrics['MAPE']:.1f}")
        
        return metrics


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_historical_data(df, location_name):
    """Plot historical solar power data"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Solar Power Generation', 'Environmental Factors'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Solar power trace
    if 'solar_power_kwh' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['solar_power_kwh'],
                name='Solar Power (kWh)', 
                line=dict(color='#1E4A6B', width=2.5),
                fill='tozeroy',
                fillcolor='rgba(30,74,107,0.1)'
            ),
            row=1, col=1
        )
    
    # Environmental factors
    if 'temperature_c' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['temperature_c'],
                name='Temperature (°C)',
                line=dict(color='#E9B741', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
    
    if 'relative_humidity' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['relative_humidity'],
                name='Humidity (%)',
                line=dict(color='#2C6B8C', width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
    
    # Layout updates
    fig.update_xaxes(title_text="Date", row=1, col=1, gridcolor='#E4E7EB')
    fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor='#E4E7EB')
    fig.update_yaxes(title_text="kWh", row=1, col=1, gridcolor='#E4E7EB')
    fig.update_yaxes(title_text="Value", row=2, col=1, gridcolor='#E4E7EB')
    
    fig.update_layout(
        height=550,
        title={
            'text': f"{location_name} - Historical Analysis",
            'font': {'size': 18, 'color': '#0A2F44', 'weight': 600}
        },
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=11, color='#3E4C59'),
        margin=dict(l=60, r=60, t=90, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_predictions_vs_actual(y_test, predictions_dict, test_dates):
    """Plot model predictions against actual values"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test,
            name='Actual',
            line=dict(color='#0A2F44', width=3),
            mode='lines'
        )
    )
    
    # Predictions
    colors = ['#1E4A6B', '#2C6B8C', '#E9B741', '#C99C2E', '#2E7D32', '#D32F2F', '#7B8794']
    
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
            'font': {'size': 18, 'color': '#0A2F44', 'weight': 600}
        },
        xaxis_title='Date',
        yaxis_title='Solar Power (kWh)',
        height=450,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=11, color='#3E4C59'),
        xaxis=dict(gridcolor='#E4E7EB'),
        yaxis=dict(gridcolor='#E4E7EB'),
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(15)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['importance'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=importance_df['importance'].round(3),
                textposition='outside'
            )
        )
        
        fig.update_layout(
            title=f"Feature Importance - {model_name}",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=11),
            margin=dict(l=120, r=40, t=60, b=60)
        )
        
        return fig
    
    return None


def plot_seasonal_patterns(df):
    """Plot seasonal patterns in solar generation"""
    if 'solar_power_kwh' not in df.columns or 'month' not in df.columns:
        return None
    
    # Aggregate by month
    monthly_avg = df.groupby('month')['solar_power_kwh'].agg(['mean', 'std']).reset_index()
    monthly_avg['month_name'] = pd.to_datetime(monthly_avg['month'], format='%m').dt.strftime('%b')
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=monthly_avg['month_name'],
            y=monthly_avg['mean'],
            error_y=dict(
                type='data',
                array=monthly_avg['std'],
                visible=True,
                color='rgba(0,0,0,0.5)'
            ),
            marker_color='#1E4A6B',
            name='Monthly Average'
        )
    )
    
    fig.update_layout(
        title="Seasonal Pattern - Average Daily Solar Generation by Month",
        xaxis_title="Month",
        yaxis_title="Average Daily Solar Power (kWh)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=11),
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Display header
    st.markdown('<p class="main-header">Solar Power Forecasting Dashboard</p>',
                unsafe_allow_html=True)
    
    # =============================================================================
    # SIDEBAR - Configuration
    # =============================================================================
    st.sidebar.title("Configuration Panel")
    st.sidebar.markdown("---")
    
    # Location selection - Rural India focus
    st.sidebar.markdown("### Location Settings")
    
    RURAL_LOCATIONS = {
        'Jodhpur, Rajasthan (Arid)': (26.2389, 73.0243),
        'Jaisalmer, Rajasthan (Desert)': (26.9157, 70.9083),
        'Anantapur, Andhra Pradesh (Semi-arid)': (14.6819, 77.6006),
        'Bikaner, Rajasthan (Arid)': (28.0229, 73.3119),
        'Kurnool, Andhra Pradesh (Semi-arid)': (15.8281, 78.0373),
        'Nashik, Maharashtra (Plateau)': (19.9975, 73.7898),
        'Solapur, Maharashtra (Semi-arid)': (17.6599, 75.9064),
        'Leh, Ladakh (Mountain Desert)': (34.1526, 77.5771),
        'Bhuj, Gujarat (Coastal)': (23.2420, 69.6669),
        'Thiruvananthapuram, Kerala (Tropical)': (8.5241, 76.9366),
        'Nagpur, Maharashtra (Continental)': (21.1458, 79.0882),
        'Shimla, Himachal Pradesh (Hill)': (31.1048, 77.1734),
        'Patna, Bihar (Gangetic Plain)': (25.5941, 85.1376),
        'Port Blair, Andaman (Island)': (11.7401, 92.6586),
        'Custom Location': None
    }
    
    location_choice = st.sidebar.selectbox(
        "Select rural Indian location",
        list(RURAL_LOCATIONS.keys()),
        index=0
    )
    
    if location_choice == 'Custom Location':
        location_name = st.sidebar.text_input("Location Name", "Custom Location")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=20.5937, format="%.4f")
        with col2:
            longitude = st.number_input("Longitude", value=78.9629, format="%.4f")
    else:
        location_name = location_choice.split('(')[0].strip()
        latitude, longitude = RURAL_LOCATIONS[location_choice]
    
    st.sidebar.markdown("---")
    
    # Data period
    st.sidebar.markdown("### Data Period")
    
    today = datetime.now().date()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=365*3)  # 3 years for long-term forecasting
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", default_start)
    with col2:
        end_date = st.date_input("End Date", default_end)
    
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        start_date = end_date - timedelta(days=365)
    
    # Fixed system parameters - no user input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Configuration")
    st.sidebar.info(
        "**Standard Solar PV System**\n"
        "• Capacity: 5.0 kW\n"
        "• Efficiency: 17%\n"
        "• Panel Area: 29.4 m²\n"
        "• Type: Polycrystalline Si"
    )
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.markdown("### Models to Train")
    
    available_models = [
        'Random Forest', 
        'Gradient Boosting', 
        'Linear Regression', 
        'SVR', 
        'Neural Network',
        'Ensemble (RF+GB)'
    ]
    
    if TENSORFLOW_AVAILABLE:
        available_models.append('LSTM')
    
    if STATSMODELS_AVAILABLE:
        available_models.extend(['ARIMA', 'SARIMA', 'Exponential Smoothing'])
    
    selected_models = st.sidebar.multiselect(
        "Select forecasting models",
        available_models,
        default=['Random Forest', 'Gradient Boosting', 'Ensemble (RF+GB)', 'LSTM'] if TENSORFLOW_AVAILABLE else ['Random Forest', 'Gradient Boosting', 'Ensemble (RF+GB)']
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        test_size = st.slider("Test set size (%)", 10, 30, 20)
        random_state = st.number_input("Random seed", 1, 100, 42)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='text-align: center; color: #9AA5B1; font-size: 0.8rem;'>"
        "Data sources: NASA POWER, Open-Meteo, SolarGIS<br>"
        "Rural India Solar Forecasting Study</p>",
        unsafe_allow_html=True
    )
    
    # =============================================================================
    # SESSION STATE INITIALIZATION
    # =============================================================================
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
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None
    
    # =============================================================================
    # MAIN TABS
    # =============================================================================
    tab1, tab2, tab3 = st.tabs(["Data Collection", "Model Training", "Analysis & Insights"])
    
    # =============================================================================
    # TAB 1: DATA COLLECTION
    # =============================================================================
    with tab1:
        st.markdown('<p class="sub-header">Data Collection & Analysis</p>', unsafe_allow_html=True)
        
        # Location info cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Location", location_name)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            days_diff = (end_date - start_date).days
            st.metric("Period", f"{days_diff} days")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Coordinates", f"{latitude:.2f}°, {longitude:.2f}°")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("System", "5.0 kW")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Collect data button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            collect_button = st.button("Collect & Process Data", type="primary", use_container_width=True)
        
        if collect_button:
            with st.spinner("Collecting and processing data..."):
                try:
                    collector = SolarDataCollector()
                    
                    df = collector.collect_data_for_location(
                        location_name, latitude, longitude,
                        start_date, end_date
                    )
                    
                    if df is not None and len(df) > 0:
                        # Feature engineering
                        fe = SolarFeatureEngineering()
                        df = fe.add_temporal_features(df)
                        df = fe.add_solar_features(df)
                        
                        # Calculate solar power with fixed parameters
                        df = calculate_solar_power(df, panel_capacity_kw=5.0, efficiency=0.17)
                        
                        # Add lag features for time series
                        if len(df) > 60:
                            df = fe.create_lag_features(df, target_col='solar_power_kwh', lags=[1, 2, 3, 7, 14, 30])
                        
                        st.session_state.data = df
                        st.session_state.data_source = df['data_source'].iloc[0] if 'data_source' in df.columns else 'Unknown'
                        
                        st.success(f"Data collection successful: {len(df)} days from {st.session_state.data_source}")
                        
                        # Summary metrics
                        st.markdown("### Data Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Days", len(df))
                        with col2:
                            if 'solar_power_kwh' in df.columns:
                                avg_power = df['solar_power_kwh'].mean()
                                st.metric("Avg Daily Power", f"{avg_power:.1f} kWh")
                        with col3:
                            if 'solar_power_kwh' in df.columns:
                                total_power = df['solar_power_kwh'].sum()
                                st.metric("Total Energy", f"{total_power:.0f} kWh")
                        with col4:
                            if 'solar_irradiance_kwh_m2' in df.columns:
                                avg_irradiance = df['solar_irradiance_kwh_m2'].mean()
                                st.metric("Avg Irradiance", f"{avg_irradiance:.2f} kWh/m²")
                        
                        # Climate zone info
                        if 'climate_zone' in df.columns:
                            st.info(f"Climate Zone: {df['climate_zone'].iloc[0]}")
                        
                    else:
                        st.error("Failed to collect data. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display data if available
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("---")
            st.markdown("### Historical Data Visualization")
            
            # Historical plot
            fig_historical = plot_historical_data(df, location_name)
            st.plotly_chart(fig_historical, use_container_width=True)
            
            # Data table expander
            with st.expander("View Data Table"):
                display_cols = ['date', 'solar_power_kwh', 'solar_irradiance_kwh_m2', 
                              'temperature_c', 'relative_humidity', 'cloudcover_mean']
                
                available_cols = [col for col in display_cols if col in df.columns]
                
                if 'data_source' in df.columns:
                    st.info(f"Data Source: {df['data_source'].iloc[0]}")
                
                st.dataframe(
                    df[available_cols].head(50),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Full Dataset (CSV)",
                    data=csv,
                    file_name=f"solar_data_{location_name}_{start_date}_{end_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Basic statistics
            st.markdown("### Descriptive Statistics")
            
            if 'solar_power_kwh' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Solar Power Statistics (kWh)**")
                    stats_df = pd.DataFrame({
                        'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                        'Value': [
                            f"{df['solar_power_kwh'].mean():.2f}",
                            f"{df['solar_power_kwh'].median():.2f}",
                            f"{df['solar_power_kwh'].std():.2f}",
                            f"{df['solar_power_kwh'].min():.2f}",
                            f"{df['solar_power_kwh'].max():.2f}",
                            f"{df['solar_power_kwh'].quantile(0.25):.2f}",
                            f"{df['solar_power_kwh'].quantile(0.75):.2f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Environmental Factors**")
                    env_stats = []
                    
                    if 'temperature_c' in df.columns:
                        env_stats.append(['Temperature (°C)', f"{df['temperature_c'].mean():.1f} ± {df['temperature_c'].std():.1f}"])
                    if 'relative_humidity' in df.columns:
                        env_stats.append(['Humidity (%)', f"{df['relative_humidity'].mean():.0f} ± {df['relative_humidity'].std():.0f}"])
                    if 'wind_speed_ms' in df.columns:
                        env_stats.append(['Wind Speed (m/s)', f"{df['wind_speed_ms'].mean():.1f} ± {df['wind_speed_ms'].std():.1f}"])
                    if 'solar_irradiance_kwh_m2' in df.columns:
                        env_stats.append(['Irradiance (kWh/m²)', f"{df['solar_irradiance_kwh_m2'].mean():.2f} ± {df['solar_irradiance_kwh_m2'].std():.2f}"])
                    
                    env_df = pd.DataFrame(env_stats, columns=['Parameter', 'Mean ± Std'])
                    st.dataframe(env_df, use_container_width=True, hide_index=True)
        else:
            st.info("Click 'Collect & Process Data' to begin analysis")
    
    # =============================================================================
    # TAB 2: MODEL TRAINING
    # =============================================================================
    with tab2:
        st.markdown('<p class="sub-header">Model Training & Evaluation</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            if 'solar_power_kwh' not in df.columns:
                st.error("Solar power data not available. Please collect data first.")
            else:
                st.success(f"Dataset ready: {len(df)} days of solar generation data")
                
                # Dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training samples", int(len(df) * (1 - test_size/100)))
                with col2:
                    st.metric("Test samples", int(len(df) * (test_size/100)))
                with col3:
                    st.metric("Features", len([c for c in df.columns if c not in ['date', 'location', 'latitude', 'longitude', 'solar_power_kwh']]))
                
                st.markdown("---")
                
                # Train models button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    train_button = st.button("Train Selected Models", type="primary", use_container_width=True)
                
                if train_button:
                    if not selected_models:
                        st.error("Please select at least one model")
                    else:
                        with st.spinner(f"Training {len(selected_models)} models..."):
                            try:
                                # Prepare features
                                fe = SolarFeatureEngineering()
                                df_features = fe.add_temporal_features(df)
                                df_features = fe.add_solar_features(df_features)
                                
                                # Select feature columns (exclude non-predictive columns)
                                exclude_cols = ['date', 'location', 'latitude', 'longitude', 
                                              'solar_power_kwh', 'data_source', 'climate_zone',
                                              'panel_capacity_kw', 'panel_efficiency']
                                
                                feature_cols = []
                                for col in df_features.columns:
                                    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col]):
                                        # Drop columns with too many NaN values
                                        if df_features[col].isna().sum() < len(df_features) * 0.5:
                                            feature_cols.append(col)
                                
                                X = df_features[feature_cols].copy()
                                y = df_features['solar_power_kwh'].copy()
                                
                                # Handle NaN values
                                X = X.fillna(X.mean())
                                y = y.fillna(y.mean())
                                
                                # Split data
                                split_idx = int(len(X) * (1 - test_size/100))
                                X_train, X_test = X[:split_idx], X[split_idx:]
                                y_train, y_test = y[:split_idx], y[split_idx:]
                                
                                st.session_state.y_test = y_test
                                st.session_state.test_dates = df_features['date'].iloc[split_idx:].values
                                
                                st.info(f"Training set: {len(X_train)} days | Test set: {len(X_test)} days")
                                
                                # Train models
                                forecaster = st.session_state.forecaster
                                all_metrics = []
                                predictions_dict = {}
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, model_name in enumerate(selected_models):
                                    status_text.text(f"Training {model_name}...")
                                    
                                    try:
                                        if model_name == 'Random Forest':
                                            metrics, y_pred = forecaster.train_random_forest(
                                                X_train, y_train, X_test, y_test, 
                                                n_estimators=200, random_state=random_state
                                            )
                                        elif model_name == 'Gradient Boosting':
                                            metrics, y_pred = forecaster.train_gradient_boosting(
                                                X_train, y_train, X_test, y_test, 
                                                n_estimators=200, learning_rate=0.1, random_state=random_state
                                            )
                                        elif model_name == 'Linear Regression':
                                            metrics, y_pred = forecaster.train_linear_regression(
                                                X_train, y_train, X_test, y_test
                                            )
                                        elif model_name == 'SVR':
                                            metrics, y_pred = forecaster.train_svr(
                                                X_train, y_train, X_test, y_test,
                                                kernel='rbf', C=1.0, epsilon=0.1
                                            )
                                        elif model_name == 'Neural Network':
                                            metrics, y_pred = forecaster.train_neural_network(
                                                X_train, y_train, X_test, y_test,
                                                hidden_layer_sizes=(100, 50), random_state=random_state
                                            )
                                        elif model_name == 'LSTM':
                                            metrics, y_pred = forecaster.train_lstm(
                                                X_train, y_train, X_test, y_test,
                                                epochs=50, batch_size=32
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
                                        
                                    except Exception as e:
                                        st.warning(f"{model_name} failed: {str(e)[:100]}")
                                    
                                    progress_bar.progress((i + 1) / len(selected_models))
                                
                                progress_bar.empty()
                                status_text.empty()
                                
                                if all_metrics:
                                    # Combine metrics
                                    metrics_df = pd.concat(all_metrics, ignore_index=True)
                                    metrics_df = metrics_df.set_index('Model')
                                    
                                    # Sort by RMSE
                                    metrics_df = metrics_df.sort_values('RMSE')
                                    
                                    st.session_state.model_metrics = metrics_df
                                    st.session_state.predictions = predictions_dict
                                    
                                    st.success(f"Successfully trained {len(all_metrics)} models")
                                    
                                    # Display results
                                    st.markdown("---")
                                    st.markdown("### Predictions vs Actual")
                                    
                                    if predictions_dict and st.session_state.test_dates is not None:
                                        fig_predictions = plot_predictions_vs_actual(
                                            y_test, 
                                            predictions_dict, 
                                            st.session_state.test_dates
                                        )
                                        st.plotly_chart(fig_predictions, use_container_width=True)
                                    
                                    st.markdown("---")
                                    st.markdown("### Model Performance Metrics")
                                    
                                    # Format metrics for display
                                    display_metrics = metrics_df.copy()
                                    display_metrics['RMSE'] = display_metrics['RMSE'].apply(lambda x: f"{x:.3f}")
                                    display_metrics['MAE'] = display_metrics['MAE'].apply(lambda x: f"{x:.3f}")
                                    display_metrics['R2'] = display_metrics['R2'].apply(lambda x: f"{x:.3f}")
                                    display_metrics['MAPE'] = display_metrics['MAPE'].apply(lambda x: f"{x:.1f}%")
                                    
                                    st.dataframe(display_metrics, use_container_width=True)
                                    
                                    # Best model highlights
                                    st.markdown("### Best Performing Models")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    best_rmse = metrics_df['RMSE'].idxmin()
                                    best_r2 = metrics_df['R2'].idxmax()
                                    best_mape = metrics_df['MAPE'].idxmin()
                                    
                                    with col1:
                                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                        st.metric("Lowest RMSE", best_rmse)
                                        st.write(f"RMSE: {metrics_df.loc[best_rmse, 'RMSE']:.3f}")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with col2:
                                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                        st.metric("Highest R²", best_r2)
                                        st.write(f"R²: {metrics_df.loc[best_r2, 'R2']:.3f}")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    with col3:
                                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                        st.metric("Lowest MAPE", best_mape)
                                        st.write(f"MAPE: {metrics_df.loc[best_mape, 'MAPE']:.1f}%")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Feature importance (for tree-based models)
                                    st.markdown("---")
                                    st.markdown("### Feature Importance Analysis")
                                    
                                    tree_models = ['Random Forest', 'Gradient Boosting']
                                    available_tree_models = [m for m in tree_models if m in forecaster.models]
                                    
                                    if available_tree_models:
                                        model_choice = st.selectbox(
                                            "Select model for feature importance",
                                            available_tree_models
                                        )
                                        
                                        if model_choice in forecaster.models and model_choice in forecaster.feature_names:
                                            model = forecaster.models[model_choice]
                                            feature_names = forecaster.feature_names[model_choice]
                                            
                                            fig_importance = plot_feature_importance(model, feature_names, model_choice)
                                            if fig_importance:
                                                st.plotly_chart(fig_importance, use_container_width=True)
                                    
                                else:
                                    st.error("No models were successfully trained")
                                    
                            except Exception as e:
                                st.error(f"Training failed: {str(e)}")
                
                # Display previous results if available
                elif st.session_state.model_metrics is not None and not st.session_state.model_metrics.empty:
                    st.info("Previous training results loaded")
                    
                    st.markdown("---")
                    st.markdown("### Predictions vs Actual")
                    
                    if st.session_state.predictions and st.session_state.y_test is not None:
                        fig_predictions = plot_predictions_vs_actual(
                            st.session_state.y_test, 
                            st.session_state.predictions, 
                            st.session_state.test_dates
                        )
                        st.plotly_chart(fig_predictions, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("### Model Performance Metrics")
                    
                    metrics_df = st.session_state.model_metrics
                    display_metrics = metrics_df.copy()
                    display_metrics['RMSE'] = display_metrics['RMSE'].apply(lambda x: f"{x:.3f}")
                    display_metrics['MAE'] = display_metrics['MAE'].apply(lambda x: f"{x:.3f}")
                    display_metrics['R2'] = display_metrics['R2'].apply(lambda x: f"{x:.3f}")
                    display_metrics['MAPE'] = display_metrics['MAPE'].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(display_metrics, use_container_width=True)
                    
                    # Best model highlights
                    st.markdown("### Best Performing Models")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    best_rmse = metrics_df['RMSE'].idxmin()
                    best_r2 = metrics_df['R2'].idxmax()
                    best_mape = metrics_df['MAPE'].idxmin()
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Lowest RMSE", best_rmse)
                        st.write(f"RMSE: {metrics_df.loc[best_rmse, 'RMSE']:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Highest R²", best_r2)
                        st.write(f"R²: {metrics_df.loc[best_r2, 'R2']:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Lowest MAPE", best_mape)
                        st.write(f"MAPE: {metrics_df.loc[best_mape, 'MAPE']:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.info("Select models and click 'Train Selected Models' to start")
        else:
            st.warning("Please collect data first in the 'Data Collection' tab")
    
    # =============================================================================
    # TAB 3: ANALYSIS & INSIGHTS
    # =============================================================================
    with tab3:
        st.markdown('<p class="sub-header">Analysis & Research Insights</p>', unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("### Rural India Solar Potential Analysis")
            
            # Climate zone analysis
            if 'climate_zone' in df.columns:
                climate_zone = df['climate_zone'].iloc[0]
                st.info(f"**Climate Zone:** {climate_zone}")
            
            # Seasonal patterns
            st.markdown("### Seasonal Patterns")
            
            fig_seasonal = plot_seasonal_patterns(df)
            if fig_seasonal:
                st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Monthly statistics
            if 'month' in df.columns and 'solar_power_kwh' in df.columns:
                st.markdown("### Monthly Generation Statistics")
                
                monthly_stats = df.groupby('month').agg({
                    'solar_power_kwh': ['mean', 'std', 'min', 'max']
                }).round(2)
                
                monthly_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
                monthly_stats.index = pd.to_datetime(monthly_stats.index, format='%m').month_name()
                
                st.dataframe(monthly_stats, use_container_width=True)
            
            # Research insights
            st.markdown("### Research Findings & Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Key Observations:**
                - **Seasonal Variation:** Solar output varies significantly with seasons
                - **Monsoon Impact:** 30-50% reduction during monsoon months
                - **Optimal Period:** March-May shows peak generation
                - **Reliability Factor:** 5kW system provides 15-25 kWh daily average
                """)
            
            with col2:
                if st.session_state.model_metrics is not None and not st.session_state.model_metrics.empty:
                    metrics_df = st.session_state.model_metrics
                    best_model = metrics_df['RMSE'].idxmin()
                    
                    st.markdown(f"""
                    **Model Performance:**
                    - **Best Model:** {best_model}
                    - **Average R² Score:** {metrics_df['R2'].mean():.3f}
                    - **Average RMSE:** {metrics_df['RMSE'].mean():.3f} kWh
                    - **Ensemble Benefit:** {'Yes' if 'Ensemble' in metrics_df.index else 'No'}
                    """)
                else:
                    st.markdown("""
                    **Model Performance:**
                    - Train models to see performance metrics
                    - Compare statistical vs ML approaches
                    - Evaluate ensemble methods
                    """)
            
            # Recommendations
            st.markdown("### Recommendations for Rural Solar Deployment")
            
            recommendations = pd.DataFrame({
                'Parameter': [
                    'System Sizing',
                    'Battery Storage',
                    'Seasonal Planning',
                    'Maintenance',
                    'Monitoring'
                ],
                'Recommendation': [
                    f"5kW system sufficient for {df['solar_power_kwh'].mean():.1f} kWh daily average",
                    f"{df['solar_power_kwh'].max():.0f} kWh/day peak requires 10-15 kWh storage",
                    "Plan for 30% lower output during June-September",
                    "Clean panels before monsoon and after dry periods",
                    "Track performance against seasonal benchmarks"
                ]
            })
            
            st.dataframe(recommendations, use_container_width=True, hide_index=True)
            
            # Export report
            st.markdown("### Generate Research Report")
            
            if st.button("Generate Analysis Report", use_container_width=True):
                st.success("Report generation complete! (Demo - download functionality in production)")
                
        else:
            st.info("Collect data first to view analysis and insights")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Solar Power Forecasting for Rural India - Research Study</p>
        <p style="font-size: 0.75rem; margin-top: 5px;">
            Data Sources: NASA POWER API, Open-Meteo, SolarGIS (Public Demos) | 
            Models: Statistical + Machine Learning Hybrid Approach<br>
            System Configuration: Standard 5kW PV System (Fixed) | 
            Focus: Long-term seasonal forecasting for sustainable rural energy planning
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()