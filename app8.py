"""
Solar Power Forecasting System for Rural India - Streamlit Application
Data Collection and Analysis for Solar Power Estimation
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

import time

# Page configuration
st.set_page_config(
    page_title="Solar Power Analysis - Rural India",
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
    
    def collect_data_for_location(self, location_name, latitude, longitude, start_date, end_date):
        """Intelligent data collection with automatic fallback"""
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


# ==================== SOLAR POWER CALCULATOR ====================
def calculate_solar_power_from_irradiance(df, panel_capacity_kw=None, efficiency=None):
    """
    Calculate solar power output from irradiance data
    
    Args:
        df: DataFrame with solar irradiance
        panel_capacity_kw: Optional - Installed solar panel capacity in kW
        efficiency: Optional - Panel efficiency (0-1)
    
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
    
    # AUTOMATIC CALCULATION: Use default values if not provided
    if panel_capacity_kw is None:
        panel_capacity_kw = 5.0  # Default: 5 kW system (common for rural households)
    
    if efficiency is None:
        efficiency = 0.17  # Default: 17% efficiency (standard panels)
    
    # Calculate panel area: Panel Capacity = Irradiance (1 kW/m²) × Area × Efficiency
    # So Area = Panel Capacity / (1 × Efficiency)
    panel_area = panel_capacity_kw / (1.0 * efficiency)
    
    # Daily energy generation (kWh) = Irradiance (kWh/m²) × Area (m²) × Efficiency
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
    
    # Store the parameters used for calculation
    df['panel_capacity_kw'] = panel_capacity_kw
    df['panel_efficiency'] = efficiency
    
    return df, panel_capacity_kw, efficiency


# ==================== FEATURE ENGINEERING ====================
class SolarFeatureEngineering:
    """Create features for solar power analysis"""
    
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


# ==================== VISUALIZATION FUNCTIONS ====================
def plot_historical_data(df, location_name, panel_capacity, efficiency):
    """Plot historical solar and weather data"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'Solar Power Generation ({panel_capacity} kW, {efficiency*100:.0f}% efficient)',
                       'Solar Irradiance', 'Temperature'),
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


def plot_seasonal_patterns(df, panel_capacity, efficiency):
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
        title=f"Monthly Average Solar Power Generation ({panel_capacity} kW, {efficiency*100:.0f}% efficient)",
        xaxis_title="Month",
        yaxis_title="Average Daily Generation (kWh)",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


# ==================== MAIN STREAMLIT APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">Solar Power Analysis System for Rural India</p>',
                unsafe_allow_html=True)
    
    project_description = """
    <div class="info-box">
    <b>Project Objective:</b> Analyze solar irradiance data and estimate solar power generation 
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
    
    # Solar System Configuration - OPTIONAL (for customization)
    st.sidebar.subheader("Solar System Configuration (Optional)")
    st.sidebar.markdown("""
    <div class="info-box" style="font-size: 0.9rem; padding: 8px; margin-bottom: 10px;">
    <b>Defaults:</b> 5 kW system, 17% efficiency<br>
    Adjust these to see how different systems would perform
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        panel_capacity = st.number_input("Panel Capacity (kW)", 
                                        value=5.0, 
                                        min_value=0.1, 
                                        max_value=100.0, 
                                        step=0.5,
                                        help="Total installed power of solar system")
    with col2:
        panel_efficiency = st.slider("Panel Efficiency (%)", 
                                    10, 25, 17, 
                                    help="How efficiently panels convert sunlight to electricity") / 100
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'calculated_power' not in st.session_state:
        st.session_state.calculated_power = None
    if 'used_capacity' not in st.session_state:
        st.session_state.used_capacity = None
    if 'used_efficiency' not in st.session_state:
        st.session_state.used_efficiency = None
    
    # Main Content Tabs
    tab1, tab2 = st.tabs([
        "Data Collection & Analysis", 
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
            st.metric("Solar System", f"{panel_capacity} kW, {panel_efficiency*100:.0f}%")
        
        # Data Collection Button
        if st.button("Collect and Analyze Solar Data", type="primary", width='stretch'):
            with st.spinner("Collecting and analyzing data..."):
                try:
                    collector = SolarDataCollector()
                    
                    # Collect data
                    df = collector.collect_data_for_location(
                        location_name, latitude, longitude,
                        start_date, end_date
                    )
                    
                    if df is not None and len(df) > 0:
                        # Add temporal and solar features
                        fe = SolarFeatureEngineering()
                        df = fe.add_temporal_features(df)
                        df = fe.add_solar_features(df)
                        
                        # AUTOMATICALLY calculate solar power using provided/default values
                        df_with_power, used_capacity, used_efficiency = calculate_solar_power_from_irradiance(
                            df, panel_capacity, panel_efficiency
                        )
                        
                        # Store in session state
                        st.session_state.data = df_with_power
                        st.session_state.calculated_power = True
                        st.session_state.used_capacity = used_capacity
                        st.session_state.used_efficiency = used_efficiency
                        
                        # Display success metrics
                        st.success(f"Successfully collected and analyzed {len(df)} days of data")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Days", len(df))
                        with col2:
                            if 'solar_power_kwh' in df_with_power.columns:
                                avg_power = df_with_power['solar_power_kwh'].mean()
                                st.metric("Avg Daily Power", f"{avg_power:.1f} kWh")
                            else:
                                st.metric("Avg Daily", "N/A")
                        with col3:
                            if 'solar_power_kwh' in df_with_power.columns:
                                total_power = df_with_power['solar_power_kwh'].sum()
                                st.metric("Total Energy", f"{total_power:.0f} kWh")
                            else:
                                st.metric("Total Energy", "N/A")
                        with col4:
                            # Calculate average irradiance correctly
                            irradiance_col = None
                            if 'solar_irradiance_kwh_m2' in df_with_power.columns:
                                irradiance_col = 'solar_irradiance_kwh_m2'
                            elif 'estimated_irradiance' in df_with_power.columns:
                                irradiance_col = 'estimated_irradiance'
                            
                            if irradiance_col:
                                # Ensure irradiance is non-negative before calculating average
                                df_with_power[irradiance_col] = df_with_power[irradiance_col].clip(lower=0)
                                avg_irrad = df_with_power[irradiance_col].mean()
                                st.metric("Avg Irradiance", f"{avg_irrad:.2f} kWh/m²")
                            else:
                                st.metric("Avg Irradiance", "N/A")
                        
                        # System performance summary
                        st.markdown("---")
                        st.subheader("Solar System Performance Summary")
                        
                        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                        with perf_col1:
                            if 'solar_power_kwh' in df_with_power.columns:
                                max_daily = df_with_power['solar_power_kwh'].max()
                                st.metric("Peak Daily", f"{max_daily:.1f} kWh")
                        with perf_col2:
                            if 'solar_power_kwh' in df_with_power.columns:
                                min_daily = df_with_power['solar_power_kwh'].min()
                                st.metric("Lowest Daily", f"{min_daily:.1f} kWh")
                        with perf_col3:
                            if 'solar_power_kwh' in df_with_power.columns:
                                capacity_factor = (df_with_power['solar_power_kwh'].mean() / (used_capacity * 24)) * 100
                                st.metric("Capacity Factor", f"{capacity_factor:.1f}%")
                        with perf_col4:
                            if 'solar_power_kwh' in df_with_power.columns:
                                annual_energy = df_with_power['solar_power_kwh'].sum() * (365 / len(df_with_power))
                                st.metric("Annual Estimate", f"{annual_energy:,.0f} kWh")
                        
                        # Insights box
                        insights = f"""
                        <div class="info-box">
                        <b>System Insights:</b><br><br>
                        • <b>System Size:</b> {used_capacity} kW, {used_efficiency*100:.0f}% efficient panels<br>
                        • <b>Panel Area:</b> {(used_capacity / (1.0 * used_efficiency)):.1f} m²<br>
                        • <b>Daily Average:</b> {avg_power:.1f} kWh/day<br>
                        • <b>Annual Estimate:</b> {annual_energy:,.0f} kWh/year<br>
                        • <b>System Utilization:</b> Equivalent to {avg_power/(used_capacity*8)*100:.1f}% of maximum potential
                        </div>
                        """
                        st.markdown(insights, unsafe_allow_html=True)
                        
                    else:
                        st.error("Failed to collect data. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display data if available
        if st.session_state.data is not None and st.session_state.calculated_power:
            df = st.session_state.data
            used_capacity = st.session_state.used_capacity
            used_efficiency = st.session_state.used_efficiency
            
            # Data Visualization
            st.markdown("---")
            st.subheader("Data Visualization")
            
            # Time series plot
            fig_historical = plot_historical_data(df, location_name, used_capacity, used_efficiency)
            st.plotly_chart(fig_historical, width='stretch')
            
            # Seasonal patterns
            fig_seasonal = plot_seasonal_patterns(df, used_capacity, used_efficiency)
            if fig_seasonal:
                st.plotly_chart(fig_seasonal, width='stretch')
            
            # Data table
            with st.expander("View Data Table"):
                # Create list of columns to display
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
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"solar_data_{location_name}_{used_capacity}kW_{int(used_efficiency*100)}percent.csv",
                mime="text/csv",
                width='stretch'
            )
        elif st.session_state.data is not None:
            st.info("Data collected. Click 'Collect and Analyze Solar Data' to calculate solar power generation.")
        else:
            st.info("Click 'Collect and Analyze Solar Data' to start data collection and analysis")
    
    # ==================== TAB 2: DOCUMENTATION ====================
    with tab2:
        st.markdown('<p class="sub-header">Documentation & Methodology</p>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Project Overview
        
        **Title:** Solar Power Analysis System for Rural India
        
        **Objective:** Analyze solar irradiance data and estimate solar power generation 
        in rural India using weather patterns and solar irradiance data.
        
        ---
        
        ## How It Works
        
        ### Automatic Solar Power Calculation:
        1. **Collects Solar Irradiance Data** (kWh/m²/day) from NASA POWER API
        2. **Automatically Calculates** solar power generation using:
           - **Default:** 5 kW system with 17% efficiency (standard rural installation)
           - **Custom:** Your specified values from sidebar
        3. **Applies Real-World Factors:**
           - Temperature efficiency effects
           - Cloud cover reduction
           - Monsoon season adjustments
        
        ### Calculation Formula:
        ```
        Daily Energy (kWh) = Solar Irradiance (kWh/m²) × Panel Area (m²) × Efficiency
        Panel Area = Panel Capacity (kW) / (1 kW/m² × Efficiency)
        ```
        
        ### Example Calculation:
        - **Location:** Jodhpur, Rajasthan
        - **Irradiance:** 5.5 kWh/m²/day (average)
        - **System:** 5 kW, 17% efficient
        - **Panel Area:** 5 / (1 × 0.17) = 29.4 m²
        - **Daily Energy:** 5.5 × 29.4 × 0.17 = **27.5 kWh/day**
        
        ---
        
        ## Application Areas
        
        ### 1. Rural Solar Planning
        - System sizing for households, schools, clinics
        - Energy production estimation
        - Battery storage planning
        
        ### 2. Policy Development
        - Regional solar potential assessment
        - Investment planning guidance
        - Energy access strategy
        
        ### 3. Research & Education
        - Solar resource assessment
        - Technology performance comparison
        - Seasonal pattern analysis
        
        ---
        
        ## How to Use This Tool
        
        ### Step 1: Configuration
        1. **Select Location** from dropdown or enter custom coordinates
        2. **Set Date Range** for historical data analysis
        3. **(Optional)** Adjust solar system parameters in sidebar
        
        ### Step 2: Data Collection & Analysis
        1. Click **"Collect and Analyze Solar Data"**
        2. System automatically:
           - Fetches historical solar/weather data
           - Calculates solar power generation
           - Shows performance metrics
           - Displays visualizations
        
        ### Step 3: Interpretation
        1. Review **daily, monthly, and seasonal patterns**
        2. Analyze **system performance metrics**
        3. Download data for **further analysis**
        
        ---
        
        ## Technical Notes
        
        ### Data Sources:
        - **Primary:** NASA POWER API for solar irradiance
        - **Fallback:** Realistic synthetic data based on location patterns
        - **Weather:** Temperature, humidity, cloud cover data
        
        ### Default Assumptions:
        - **Panel Capacity:** 5 kW (standard rural household system)
        - **Efficiency:** 17% (typical polycrystalline panels)
        - **Performance Factors:** Temperature, clouds, monsoon effects
        
        ### Limitations:
        - Historical data availability depends on API access
        - Does not account for local shading or installation issues
        - Daily resolution analysis only
        
        ---
        
        ## References
        
        1. NASA POWER Data Services
        2. Indian Meteorological Department
        3. Solar Energy Research for Rural Applications
        4. Renewable Energy System Design Guidelines
        
        ---
        
        ## Support
        
        **Note:** This tool provides estimates for planning purposes. 
        For actual system design, consult with solar energy professionals.
        
        Default values represent typical rural solar installations in India.
        Adjust parameters in sidebar to simulate different system configurations.
        """)


# Run the app
if __name__ == "__main__":
    main()