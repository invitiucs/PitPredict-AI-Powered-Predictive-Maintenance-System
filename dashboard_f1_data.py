"""
PitPredict Dashboard with FastF1 Real Data Integration
Run with: streamlit run dashboard_f1_data.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import time
import os

# Page configuration
st.set_page_config(
    page_title="PitPredict + FastF1 - Real F1 Data",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E10600;
        margin-bottom: 1rem;
    }
    .healthy-alert {
        background-color: #00C851;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #007E33;
        margin: 10px 0;
        color: white;
    }
    .warning-alert {
        background-color: #ffbb33;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff8800;
        margin: 10px 0;
    }
    .critical-alert {
        background-color: #ff4444;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #cc0000;
        margin: 10px 0;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def load_model():
    """Load trained model"""
    try:
        possible_files = ['pitpredict-model.pkl', 'pitpredict_model.pkl']
        
        for filename in possible_files:
            if os.path.exists(filename):
                model_data = joblib.load(filename)
                return model_data['model'], model_data['scaler'], model_data['feature_names']
        
        st.error("⚠️ Model file not found! Please run train.py first.")
        st.stop()
        
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

def engineer_features(df):
    """Apply feature engineering"""
    df = df.copy()
    
    df['vibration_temp_ratio'] = df['vibration'] / (df['temperature'] + 1)
    df['pressure_rpm_ratio'] = df['pressure'] / (df['rpm'] + 1)
    df['runtime_load_ratio'] = df['runtime_hours'] / (df['load_cycles'] + 1)
    df['vibration_squared'] = df['vibration'] ** 2
    df['temperature_squared'] = df['temperature'] ** 2
    df['risk_score'] = (
        (df['vibration'] / 100) * 0.3 +
        (df['temperature'] / 120) * 0.25 +
        (df['runtime_hours'] / 500) * 0.2 +
        ((100 - df['pressure']) / 100) * 0.15 +
        (df['vibration_variance'] / 15) * 0.1
    )
    
    return df

def load_f1_data():
    """Load real F1 telemetry data if available"""
    if os.path.exists('f1_telemetry_pitpredict.csv'):
        return pd.read_csv('f1_telemetry_pitpredict.csv')
    return None

def create_gauge_chart(value, title, max_value, threshold_warn, threshold_crit):
    """Create gauge chart for sensor reading"""
    
    if value > threshold_crit:
        color = "red"
    elif value > threshold_warn:
        color = "orange"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold_warn], 'color': "lightgreen"},
                {'range': [threshold_warn, threshold_crit], 'color': "lightyellow"},
                {'range': [threshold_crit, max_value], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_crit
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🏎️ PitPredict + FastF1</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Real Formula 1 Telemetry Analysis</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    # Sidebar
    st.sidebar.header("⚙️ Data Source")
    
    data_mode = st.sidebar.radio(
        "Select Data Mode",
        ["Simulated Data", "Real F1 Telemetry (FastF1)"]
    )
    
    # Check if F1 data exists
    f1_data = load_f1_data()
    
    if data_mode == "Real F1 Telemetry (FastF1)":
        if f1_data is None:
            st.sidebar.warning("⚠️ No F1 telemetry data found!")
            st.sidebar.info("""
            To use real F1 data:
            1. Run `fastf1_data_loader.py`
            2. This will download and convert F1 telemetry
            3. Refresh this dashboard
            """)
            
            st.info("""
            ### 📥 How to Load Real F1 Data
            
            1. **Install FastF1:**
            ```bash
            pip install fastf1
            ```
            
            2. **Run the data loader:**
            ```bash
            python fastf1_data_loader.py
            ```
            
            3. **Refresh this page**
            
            The system will fetch real telemetry from recent F1 races!
            """)
            st.stop()
        
        st.sidebar.success(f"✅ {len(f1_data)} F1 data points loaded")
        
        # F1 data playback controls
        st.sidebar.header("🎮 Playback Controls")
        playback_speed = st.sidebar.slider("Playback Speed", 1, 50, 10)
        
        # Initialize or get playback position
        if 'f1_position' not in st.session_state:
            st.session_state.f1_position = 0
        
        # Get current reading from F1 data
        reading = f1_data.iloc[st.session_state.f1_position].to_dict()
        
        # Show F1-specific info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 F1 Data Info")
        if 'f1_speed' in reading and pd.notna(reading['f1_speed']):
            try:
                st.sidebar.metric("Speed", f"{float(reading['f1_speed']):.1f} km/h")
            except:
                st.sidebar.metric("Speed", str(reading['f1_speed']))
        
        if 'f1_throttle' in reading and pd.notna(reading['f1_throttle']):
            try:
                st.sidebar.metric("Throttle", f"{float(reading['f1_throttle']):.0f}%")
            except:
                st.sidebar.metric("Throttle", str(reading['f1_throttle']))
        
        if 'f1_brake' in reading and pd.notna(reading['f1_brake']):
            try:
                brake_val = reading['f1_brake']
                # Convert various types to number
                if isinstance(brake_val, bool):
                    brake_display = "100%" if brake_val else "0%"
                elif isinstance(brake_val, str):
                    brake_display = "100%" if brake_val.lower() == 'true' else "0%"
                else:
                    brake_display = f"{float(brake_val):.0f}%"
                st.sidebar.metric("Brake", brake_display)
            except:
                st.sidebar.metric("Brake", str(reading['f1_brake']))
        
        # Advance position
        st.session_state.f1_position = (st.session_state.f1_position + 1) % len(f1_data)
        
    else:
        # Simulated data mode
        component_name = st.sidebar.selectbox(
            "Select Component",
            ["Turbocharger", "Gearbox", "Brake System", "Suspension", "Cooling System"]
        )
        
        failure_mode = st.sidebar.checkbox("🔴 Simulate Failure Mode", value=False)
        
        # Generate simulated reading
        if failure_mode:
            vibration = np.random.normal(75, 12)
            temperature = np.random.normal(95, 8)
            rpm = np.random.normal(18000, 1200)
            pressure = np.random.normal(82, 10)
            noise_level = np.random.normal(78, 6)
        else:
            vibration = np.random.normal(45, 8)
            temperature = np.random.normal(75, 6)
            rpm = np.random.normal(15000, 800)
            pressure = np.random.normal(100, 6)
            noise_level = np.random.normal(65, 5)
        
        reading = {
            'timestamp': datetime.now(),
            'runtime_hours': np.random.uniform(200, 400),
            'vibration': max(0, vibration),
            'temperature': max(20, temperature),
            'rpm': max(0, rpm),
            'pressure': max(0, pressure),
            'noise_level': max(30, noise_level),
            'load_cycles': int(np.random.uniform(15000, 30000)),
            'vibration_variance': np.random.uniform(2, 10),
            'temp_gradient': np.random.uniform(-2, 5)
        }
    
    # Common controls
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
    alert_threshold = st.sidebar.slider("Alert Threshold (%)", 0, 100, 50)
    
    # Prepare data for prediction
    reading_df = pd.DataFrame([reading])
    reading_df = engineer_features(reading_df)
    
    # Keep only model features
    X = reading_df[feature_names]
    X_scaled = scaler.transform(X)
    
    # Predict
    failure_prob = model.predict_proba(X_scaled)[0, 1] * 100
    
    # Store in history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'timestamp': reading.get('timestamp', datetime.now()),
        'failure_prob': failure_prob,
        'vibration': reading['vibration'],
        'temperature': reading['temperature'],
        'pressure': reading['pressure']
    })
    
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]
    
    # Status alert
    st.markdown("---")
    
    component_display = "F1 Component" if data_mode == "Real F1 Telemetry (FastF1)" else component_name
    
    if failure_prob >= 70:
        st.markdown(f"""
            <div class="critical-alert">
                <h2>⚠️ CRITICAL ALERT</h2>
                <h3>{component_display} - Failure Probability: {failure_prob:.1f}%</h3>
                <p>Immediate maintenance required! Component at high risk of failure.</p>
            </div>
        """, unsafe_allow_html=True)
    elif failure_prob >= alert_threshold:
        st.markdown(f"""
            <div class="warning-alert">
                <h2>⚠️ WARNING</h2>
                <h3>{component_display} - Failure Probability: {failure_prob:.1f}%</h3>
                <p>Schedule maintenance soon. Component showing signs of degradation.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="healthy-alert">
                <h2>✅ HEALTHY</h2>
                <h3>{component_display} - Failure Probability: {failure_prob:.1f}%</h3>
                <p>Component operating within normal parameters.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("### 📊 Real-Time Sensor Readings")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vibration", f"{reading['vibration']:.1f} mm/s",
                 delta=f"{reading['vibration'] - 45:.1f}" if reading['vibration'] > 45 else None)
    
    with col2:
        st.metric("Temperature", f"{reading['temperature']:.1f}°C",
                 delta=f"{reading['temperature'] - 75:.1f}" if reading['temperature'] > 75 else None)
    
    with col3:
        st.metric("Pressure", f"{reading['pressure']:.1f} PSI",
                 delta=f"{100 - reading['pressure']:.1f}" if reading['pressure'] < 100 else None,
                 delta_color="inverse")
    
    with col4:
        st.metric("RPM", f"{reading['rpm']:.0f}",
                 delta=None)
    
    # Gauges
    st.markdown("### 🎯 Live Gauges")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = create_gauge_chart(reading['vibration'], "Vibration (mm/s)", 120, 60, 80)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gauge_chart(reading['temperature'], "Temperature (°C)", 140, 85, 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_gauge_chart(reading['rpm'], "RPM", 20000, 16000, 18000)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = create_gauge_chart(reading['pressure'], "Pressure (PSI)", 120, 85, 75)
        st.plotly_chart(fig, use_container_width=True)
    
    # Historical trends
    if len(st.session_state.history) > 1:
        st.markdown("### 📈 Historical Trends")
        history_df = pd.DataFrame(st.session_state.history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['failure_prob'],
            mode='lines+markers',
            name='Failure Probability',
            line=dict(color='red', width=3),
            fill='tozeroy'
        ))
        
        fig.add_hline(y=alert_threshold, line_dash="dash", line_color="orange",
                     annotation_text=f"Alert Threshold ({alert_threshold}%)")
        
        fig.update_layout(
            title="Failure Probability Over Time",
            xaxis_title="Time",
            yaxis_title="Failure Probability (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()