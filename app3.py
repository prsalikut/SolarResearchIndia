"""
Solar Power Forecasting System for Rural India
Enhanced UI with Light Color Theme
Author: Pranay Salikuti
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Solar Forecasting - Rural India",
    page_icon="☀️",
    layout="wide"
)

# Enhanced CSS with Light Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding: 2rem;
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 30px;
        animation: fadeIn 1s;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #764ba2;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102,126,234,0.3);
        transition: transform 0.3s;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
    }
    
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    
    .success-alert {
        background: linear-gradient(135deg, #81FBB8 0%, #28C76F 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(40,199,111,0.3);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA COLLECTOR ====================
class SolarDataCollector:
    def __init__(self):
        self.open_meteo_base = "https://archive-api.open-meteo.com/v1/archive"
        self.nasa_power_base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def get_weather_data(self, lat, lon, start, end):
        params = {
            'latitude': lat, 'longitude': lon,
            'start_date': start, 'end_date': end,
            'daily': 'temperature_2m_mean,precipitation_sum,windspeed_10m_max,'
                    'shortwave_radiation_sum,relative_humidity_2m_mean,cloudcover_mean',
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
        except Exception as e:
            st.error(f"Error: {e}")
        return None
    
    def get_solar_data(self, lat, lon, start, end):
        start_f = start.replace('-', '')
        end_f = end.replace('-', '')
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M',
            'community': 'RE',
            'longitude': lon, 'latitude': lat,
            'start': start_f, 'end': end_f,
            'format': 'JSON'
        }
        try:
            response = requests.get(self.nasa_power_base, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            if 'properties' in data:
                params_data = data['properties']['parameter']
                dates = list(params_data['ALLSKY_SFC_SW_DWN'].keys())
                df = pd.DataFrame({
                    'date': pd.to_datetime(dates, format='%Y%m%d'),
                    'solar_irradiance': list(params_data['ALLSKY_SFC_SW_DWN'].values()),
                    'clear_sky_irradiance': list(params_data['CLRSKY_SFC_SW_DWN'].values())
                })
                return df
        except Exception as e:
            st.error(f"Error: {e}")
        return None
    
    def collect(self, name, lat, lon, start, end):
        progress = st.progress(0)
        status = st.empty()
        
        status.markdown(f"<div class='info-card'>🔄 Fetching weather data...</div>", unsafe_allow_html=True)
        progress.progress(33)
        weather_df = self.get_weather_data(lat, lon, start, end)
        
        status.markdown(f"<div class='info-card'>🛰️ Fetching solar data...</div>", unsafe_allow_html=True)
        progress.progress(66)
        solar_df = self.get_solar_data(lat, lon, start, end)
        
        if weather_df is not None and solar_df is not None:
            status.markdown(f"<div class='info-card'>🔗 Merging datasets...</div>", unsafe_allow_html=True)
            progress.progress(90)
            merged = pd.merge(weather_df, solar_df, on='date', how='inner')
            merged['location'] = name
            progress.progress(100)
            status.markdown(f"<div class='success-alert'>✅ Collected {len(merged)} days of data!</div>", unsafe_allow_html=True)
            time.sleep(2)
            status.empty()
            progress.empty()
            return merged
        progress.empty()
        status.empty()
        return None

# ==================== FEATURE ENGINEERING ====================
def engineer_features(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['season'] = df['month'].apply(lambda x: 
        1 if x in [12,1,2] else 2 if x in [3,4,5] else 3 if x in [6,7,8,9] else 4)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Solar features
    df['clearness_index'] = df['solar_irradiance'] / (df['clear_sky_irradiance'] + 0.001)
    df['temp_factor'] = 1 - 0.005 * (df['temperature_2m_mean'] - 25)
    df['cloud_factor'] = 1 - (df['cloudcover_mean'] / 100) * 0.75
    
    return df

def calculate_power(df, capacity=1.0, efficiency=0.15):
    df = df.copy()
    df['solar_power'] = df['solar_irradiance'] * capacity * efficiency
    if 'temp_factor' in df.columns:
        df['solar_power'] *= df['temp_factor']
    if 'cloud_factor' in df.columns:
        df['solar_power'] *= df['cloud_factor']
    return df

# ==================== MODEL ====================
class SolarModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def train(self, X, y):
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_importance(self):
        if self.feature_names:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

# ==================== METRICS ====================
def calc_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }

# ==================== PLOTS ====================
def plot_time_series(df, name):
    fig = make_subplots(rows=3, cols=1, 
                        subplot_titles=('☀️ Solar Irradiance', '🌡️ Temperature', '☁️ Cloud Cover'),
                        vertical_spacing=0.12)
    
    fig.add_trace(go.Scatter(x=df['date'], y=df['solar_irradiance'],
                            line=dict(color='#FF6B35', width=2),
                            fill='tozeroy', fillcolor='rgba(255,107,53,0.2)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['temperature_2m_mean'],
                            line=dict(color='#F7931E', width=2),
                            fill='tozeroy', fillcolor='rgba(247,147,30,0.2)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['cloudcover_mean'],
                            line=dict(color='#667eea', width=2),
                            fill='tozeroy', fillcolor='rgba(102,126,234,0.2)'), row=3, col=1)
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="kWh/m²", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_layout(height=800, showlegend=False,
                     title_text=f"Weather Data - {name}",
                     plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_predictions(y_true, y_pred, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines',
                            name='Actual', line=dict(color='#667eea', width=3),
                            fill='tozeroy', fillcolor='rgba(102,126,234,0.2)'))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines',
                            name='Predicted', line=dict(color='#f093fb', width=3, dash='dash')))
    fig.update_layout(title="Actual vs Predicted Solar Power",
                     xaxis_title="Date", yaxis_title="Solar Power (kWh)",
                     height=500, plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
        <div class='main-title'>☀️ Solar Power Forecasting</div>
        <div class='subtitle'>for Rural India</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <b>📋 Research Project:</b> Forecasting Models for Long-Term Solar Energy Trends<br>
        <b>👨‍🔬 Author:</b> Pranay Salikuti<br>
        <b>🎯 Approach:</b> Hybrid ML + Statistical Methods for Weather-Driven Solar Forecasting
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:white; text-align:center;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
        
        locations = {
            '🏜️ Jodhpur, Rajasthan': (26.2389, 73.0243),
            '🏜️ Jaisalmer, Rajasthan': (26.9157, 70.9083),
            '☀️ Anantapur, AP': (14.6819, 77.6006),
            '📍 Custom': None
        }
        
        location = st.selectbox("Location", list(locations.keys()))
        
        if location == '📍 Custom':
            name = st.text_input("Location Name", "My Location")
            lat = st.number_input("Latitude", value=26.2389)
            lon = st.number_input("Longitude", value=73.0243)
        else:
            name = location
            lat, lon = locations[location]
            st.info(f"📍 {lat}°N, {lon}°E")
        
        st.markdown("<h3 style='color:white;'>📅 Date Range</h3>", unsafe_allow_html=True)
        start_date = st.date_input("Start", datetime.now() - timedelta(days=730))
        end_date = st.date_input("End", datetime.now() - timedelta(days=1))
        
        st.markdown("<h3 style='color:white;'>⚡ Solar Setup</h3>", unsafe_allow_html=True)
        capacity = st.number_input("Capacity (kW)", value=1.0, step=0.1)
        efficiency = st.slider("Efficiency (%)", 10, 25, 15) / 100
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🔍 Analysis", "🤖 Training", "📈 Results"])
    
    # Session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    # Tab 1: Data Collection
    with tab1:
        st.markdown("### 📊 Data Collection")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-box'><h3>{name}</h3>Location</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-box'><h3>{(end_date-start_date).days}</h3>Days</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-box'><h3>{capacity} kW</h3>Capacity</div>", unsafe_allow_html=True)
        
        if st.button("🚀 Collect Data", width='stretch'):
            collector = SolarDataCollector()
            df = collector.collect(name, lat, lon, 
                                  start_date.strftime('%Y-%m-%d'),
                                  end_date.strftime('%Y-%m-%d'))
            
            if df is not None and len(df) > 0:
                df = engineer_features(df)
                df = calculate_power(df, capacity, efficiency)
                df = df.dropna()
                st.session_state.data = df
                
                col1, col2, col3, col4 = st.columns(4)
                metrics = [
                    (col1, "Days", len(df)),
                    (col2, "Avg Power", f"{df['solar_power'].mean():.2f} kWh"),
                    (col3, "Max Power", f"{df['solar_power'].max():.2f} kWh"),
                    (col4, "Avg Temp", f"{df['temperature_2m_mean'].mean():.1f}°C")
                ]
                for col, label, value in metrics:
                    with col:
                        st.markdown(f"<div class='metric-box'><h3>{value}</h3>{label}</div>", unsafe_allow_html=True)
                
                st.dataframe(df[['date','solar_irradiance','solar_power','temperature_2m_mean']].head(10))
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv,
                                 f"solar_data_{name}.csv", "text/csv")
    
    # Tab 2: Analysis
    with tab2:
        st.markdown("### 🔍 Data Analysis")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            col1, col2, col3, col4 = st.columns(4)
            stats = [
                (col1, "Total Energy", f"{df['solar_power'].sum():.0f} kWh"),
                (col2, "Std Dev", f"{df['solar_power'].std():.2f}"),
                (col3, "Min Power", f"{df['solar_power'].min():.2f} kWh"),
                (col4, "Peak Irr.", f"{df['solar_irradiance'].max():.2f}")
            ]
            for col, label, value in stats:
                with col:
                    st.markdown(f"<div class='metric-box'><h3>{value}</h3>{label}</div>", unsafe_allow_html=True)
            
            st.plotly_chart(plot_time_series(df, name), width='stretch')
            
            st.markdown("#### 📊 Statistics")
            st.dataframe(df[['solar_irradiance','solar_power','temperature_2m_mean']].describe())
        else:
            st.warning("⚠️ Please collect data first")
    
    # Tab 3: Training
    with tab3:
        st.markdown("### 🤖 Model Training")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            exclude = ['date', 'location', 'solar_power']
            X = df[[c for c in df.columns if c not in exclude]]
            y = df['solar_power']
            
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            split_idx = int(len(X) * (1 - test_size))
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            dates_test = df['date'][split_idx:]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-box'><h3>{len(X.columns)}</h3>Features</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-box'><h3>{len(X_train)}</h3>Train</div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-box'><h3>{len(X_test)}</h3>Test</div>", unsafe_allow_html=True)
            
            if st.button("🎯 Train Model", width='stretch'):
                with st.spinner("Training..."):
                    model = SolarModel()
                    model.train(X_train, y_train)
                    
                    st.session_state.model = model
                    st.session_state.trained = True
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.dates_test = dates_test
                    
                    st.markdown("<div class='success-alert'>✅ Model trained successfully!</div>", unsafe_allow_html=True)
                    
                    importance = model.get_importance()
                    if importance is not None:
                        fig = go.Figure(go.Bar(
                            x=importance.head(10)['importance'],
                            y=importance.head(10)['feature'],
                            orientation='h',
                            marker=dict(color=importance.head(10)['importance'], 
                                      colorscale='Purples')
                        ))
                        fig.update_layout(title="Top 10 Features", height=400,
                                        plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, width='stretch')
        else:
            st.warning("⚠️ Please collect data first")
    
    # Tab 4: Results
    with tab4:
        st.markdown("### 📈 Results & Evaluation")
        
        if st.session_state.trained:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            dates_test = st.session_state.dates_test
            
            y_pred = model.predict(X_test)
            metrics = calc_metrics(y_test.values, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            metric_data = [
                (col1, "RMSE", f"{metrics['RMSE']:.4f}"),
                (col2, "MAE", f"{metrics['MAE']:.4f}"),
                (col3, "R²", f"{metrics['R²']:.4f}"),
                (col4, "MAPE", f"{metrics['MAPE']:.2f}%")
            ]
            for col, label, value in metric_data:
                with col:
                    st.markdown(f"<div class='metric-box'><h2>{value}</h2>{label}</div>", unsafe_allow_html=True)
            
            r2 = metrics['R²']
            quality = "Excellent" if r2 > 0.9 else "Very Good" if r2 > 0.8 else "Good" if r2 > 0.7 else "Fair"
            color = "#28C76F" if r2 > 0.9 else "#81FBB8" if r2 > 0.8 else "#FFA726" if r2 > 0.7 else "#FF6B6B"
            
            st.markdown(f"<div style='background:{color}; padding:20px; border-radius:15px; color:white; text-align:center; font-size:1.3rem; font-weight:600;'>Model Quality: {quality}</div>", unsafe_allow_html=True)
            
            st.plotly_chart(plot_predictions(y_test.values, y_pred, dates_test.values), width='stretch')
            
            results_df = pd.DataFrame({
                'date': dates_test.values,
                'actual': y_test.values,
                'predicted': y_pred,
                'error': y_test.values - y_pred
            })
            
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv,
                             f"predictions_{name}.csv", "text/csv")
        else:
            st.warning("⚠️ Please train the model first")

if __name__ == "__main__":
    main()
