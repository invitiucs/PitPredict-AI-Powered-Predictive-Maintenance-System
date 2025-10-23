"""
PitPredict Advanced Dashboard - Team & Driver Selection
Run with: streamlit run dashboard_advanced.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import os
import sys

# Page configuration
st.set_page_config(
    page_title="PitPredict - Advanced F1 Analysis",
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
    .team-header {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .healthy-alert { background-color: #00C851; padding: 15px; border-radius: 8px; color: white; }
    .warning-alert { background-color: #ffbb33; padding: 15px; border-radius: 8px; }
    .critical-alert { background-color: #ff4444; padding: 15px; border-radius: 8px; color: white; }
    </style>
""", unsafe_allow_html=True)

# F1 2024 Teams and Drivers
F1_TEAMS = {
    "Red Bull Racing": {
        "drivers": ["VER - Max Verstappen", "PER - Sergio Perez"],
        "color": "#1E41FF",
        "components": ["Power Unit (Honda RBPT)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Mercedes": {
        "drivers": ["HAM - Lewis Hamilton", "RUS - George Russell"],
        "color": "#00D2BE",
        "components": ["Power Unit (Mercedes)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Ferrari": {
        "drivers": ["LEC - Charles Leclerc", "SAI - Carlos Sainz"],
        "color": "#DC0000",
        "components": ["Power Unit (Ferrari)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "McLaren": {
        "drivers": ["NOR - Lando Norris", "PIA - Oscar Piastri"],
        "color": "#FF8700",
        "components": ["Power Unit (Mercedes)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Aston Martin": {
        "drivers": ["ALO - Fernando Alonso", "STR - Lance Stroll"],
        "color": "#006F62",
        "components": ["Power Unit (Mercedes)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Alpine": {
        "drivers": ["GAS - Pierre Gasly", "OCO - Esteban Ocon"],
        "color": "#0090FF",
        "components": ["Power Unit (Renault)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Williams": {
        "drivers": ["ALB - Alexander Albon", "SAR - Logan Sargeant"],
        "color": "#005AFF",
        "components": ["Power Unit (Mercedes)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "RB (AlphaTauri)": {
        "drivers": ["TSU - Yuki Tsunoda", "RIC - Daniel Ricciardo"],
        "color": "#2B4562",
        "components": ["Power Unit (Honda RBPT)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Kick Sauber": {
        "drivers": ["BOT - Valtteri Bottas", "ZHO - Zhou Guanyu"],
        "color": "#900000",
        "components": ["Power Unit (Ferrari)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    },
    "Haas": {
        "drivers": ["MAG - Kevin Magnussen", "HUL - Nico Hulkenberg"],
        "color": "#FFFFFF",
        "components": ["Power Unit (Ferrari)", "Gearbox", "Suspension", "Brake System", "Cooling System"]
    }
}

# Grand Prix list
GRAND_PRIX = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]

def load_model():
    """Load trained model"""
    try:
        possible_files = ['pitpredict-model.pkl', 'pitpredict_model.pkl']
        for filename in possible_files:
            if os.path.exists(filename):
                model_data = joblib.load(filename)
                return model_data['model'], model_data['scaler'], model_data['feature_names']
        st.error("⚠️ Model file not found!")
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

def load_f1_telemetry(driver_code, race, year=2024):
    """Load F1 telemetry for specific driver and race"""
    try:
        # Import here to avoid issues if fastf1 not installed
        sys.path.insert(0, os.path.dirname(__file__))
        from fastf1_data_loader import FastF1DataLoader
        
        loader = FastF1DataLoader()
        
        with st.spinner(f"📥 Loading {year} {race} data for {driver_code}..."):
            success = loader.load_session(year, race, 'R')
            
            if not success and year == 2024:
                st.info(f"2024 {race} not available, trying 2023...")
                success = loader.load_session(2023, race, 'R')
            
            if success:
                telemetry = loader.get_driver_telemetry(driver_code)
                if telemetry is not None:
                    data = loader.convert_to_pitpredict_format(telemetry)
                    return data, loader
        
        return None, None
    except ImportError:
        st.error("⚠️ FastF1 not installed. Run: pip install fastf1")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return None, None

def create_gauge_chart(value, title, max_value, threshold_warn, threshold_crit):
    """Create gauge chart"""
    color = "red" if value > threshold_crit else ("orange" if value > threshold_warn else "green")
    
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
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold_crit}
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<div class="main-header">🏎️ PitPredict - Advanced F1 Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Team & Driver Specific Vehicle Analysis</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    # Sidebar - Team Selection
    st.sidebar.header("🏁 Select Team & Driver")
    
    selected_team = st.sidebar.selectbox(
        "Choose F1 Team",
        list(F1_TEAMS.keys())
    )
    
    team_info = F1_TEAMS[selected_team]
    
    # Display team color
    st.sidebar.markdown(f"""
        <div style="background-color: {team_info['color']}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">
            {selected_team}
        </div>
    """, unsafe_allow_html=True)
    
    # Driver selection
    selected_driver_full = st.sidebar.selectbox(
        "Choose Driver",
        team_info["drivers"]
    )
    
    driver_code = selected_driver_full.split(" - ")[0]
    driver_name = selected_driver_full.split(" - ")[1]
    
    # Component selection
    selected_component = st.sidebar.selectbox(
        "Select Component to Monitor",
        team_info["components"]
    )
    
    st.sidebar.markdown("---")
    
    # Data source selection
    data_mode = st.sidebar.radio(
        "Data Source",
        ["Simulated Data", "Real F1 Race Data"]
    )
    
    if data_mode == "Real F1 Race Data":
        st.sidebar.markdown("### 🏁 Race Selection")
        
        selected_race = st.sidebar.selectbox("Grand Prix", GRAND_PRIX)
        selected_year = st.sidebar.selectbox("Year", [2024, 2023, 2022, 2021])
        
        if st.sidebar.button("📥 Load Race Data"):
            telemetry_data, loader = load_f1_telemetry(driver_code, selected_race, selected_year)
            
            if telemetry_data is not None:
                st.session_state['f1_data'] = telemetry_data
                st.session_state['loader'] = loader
                st.session_state['driver_name'] = driver_name
                st.session_state['driver_code'] = driver_code
                st.session_state['race'] = selected_race
                st.session_state['year'] = selected_year
                st.success(f"✅ Loaded {len(telemetry_data)} data points!")
            else:
                st.warning("⚠️ Could not load race data. Using simulated data.")
        
        # Use loaded data if available
        if 'f1_data' in st.session_state and st.session_state.get('driver_code') == driver_code:
            if 'f1_position' not in st.session_state:
                st.session_state.f1_position = 0
            
            reading = st.session_state['f1_data'].iloc[st.session_state.f1_position].to_dict()
            st.session_state.f1_position = (st.session_state.f1_position + 1) % len(st.session_state['f1_data'])
            
            # Show race info
            st.sidebar.markdown("---")
            st.sidebar.success(f"📊 **{st.session_state['year']} {st.session_state['race']}**")
            st.sidebar.info(f"👤 **{st.session_state['driver_name']}**")
            
        else:
            st.sidebar.info("👆 Click 'Load Race Data' to analyze real telemetry")
            data_mode = "Simulated Data"
    
    # Generate simulated data if not using real data
    if data_mode == "Simulated Data":
        failure_mode = st.sidebar.checkbox("🔴 Simulate Component Failure", value=False)
        
        if failure_mode:
            vibration = np.random.normal(75, 12)
            temperature = np.random.normal(95, 8)
            rpm = np.random.normal(18000, 1200)
            pressure = np.random.normal(82, 10)
        else:
            vibration = np.random.normal(45, 8)
            temperature = np.random.normal(75, 6)
            rpm = np.random.normal(15000, 800)
            pressure = np.random.normal(100, 6)
        
        reading = {
            'timestamp': datetime.now(),
            'runtime_hours': np.random.uniform(200, 400),
            'vibration': max(0, vibration),
            'temperature': max(20, temperature),
            'rpm': max(0, rpm),
            'pressure': max(0, pressure),
            'noise_level': max(30, np.random.normal(65, 5)),
            'load_cycles': int(np.random.uniform(15000, 30000)),
            'vibration_variance': np.random.uniform(2, 10),
            'temp_gradient': np.random.uniform(-2, 5)
        }
    
    # Common controls
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 1, 10, 3)
    alert_threshold = st.sidebar.slider("Alert Threshold (%)", 0, 100, 50)
    
    # Prepare data for prediction
    reading_df = pd.DataFrame([reading])
    reading_df = engineer_features(reading_df)
    X = reading_df[feature_names]
    X_scaled = scaler.transform(X)
    failure_prob = model.predict_proba(X_scaled)[0, 1] * 100
    
    # Store history
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
    
    # Main display
    st.markdown("---")
    
    # Team header with component
    st.markdown(f"""
        <div style="background-color: {team_info['color']}; padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="margin: 0;">🏎️ {selected_team} - {driver_name}</h2>
            <h3 style="margin: 5px 0 0 0;">Component: {selected_component}</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Status alert
    if failure_prob >= 70:
        st.markdown(f"""
            <div class="critical-alert">
                <h2>⚠️ CRITICAL ALERT</h2>
                <h3>Failure Probability: {failure_prob:.1f}%</h3>
                <p>🔴 <strong>IMMEDIATE ACTION REQUIRED</strong></p>
                <p>Replace {selected_component} before next session!</p>
            </div>
        """, unsafe_allow_html=True)
    elif failure_prob >= alert_threshold:
        st.markdown(f"""
            <div class="warning-alert">
                <h2>⚠️ WARNING</h2>
                <h3>Failure Probability: {failure_prob:.1f}%</h3>
                <p>🟡 <strong>MAINTENANCE RECOMMENDED</strong></p>
                <p>Schedule inspection of {selected_component} within 24 hours.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="healthy-alert">
                <h2>✅ HEALTHY</h2>
                <h3>Failure Probability: {failure_prob:.1f}%</h3>
                <p>🟢 <strong>OPERATING NORMALLY</strong></p>
                <p>{selected_component} is within safe parameters.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("### 📊 Live Sensor Readings")
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
        st.metric("RPM", f"{reading['rpm']:.0f}")
    
    # Gauges
    st.markdown("### 🎯 Live Component Gauges")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.plotly_chart(create_gauge_chart(reading['vibration'], "Vibration", 120, 60, 80), use_container_width=True)
    with col2:
        st.plotly_chart(create_gauge_chart(reading['temperature'], "Temperature", 140, 85, 100), use_container_width=True)
    with col3:
        st.plotly_chart(create_gauge_chart(reading['rpm'], "RPM", 20000, 16000, 18000), use_container_width=True)
    with col4:
        st.plotly_chart(create_gauge_chart(reading['pressure'], "Pressure", 120, 85, 75), use_container_width=True)
    
    # Historical trends
    if len(st.session_state.history) > 1:
        st.markdown("### 📈 Failure Probability Trend")
        history_df = pd.DataFrame(st.session_state.history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df['failure_prob'],
            mode='lines+markers',
            name='Failure Probability',
            line=dict(color=team_info['color'], width=3),
            fill='tozeroy',
            fillcolor=f"rgba{tuple(list(bytes.fromhex(team_info['color'][1:])) + [0.3])}"
        ))
        
        fig.add_hline(y=alert_threshold, line_dash="dash", line_color="orange",
                     annotation_text=f"Alert ({alert_threshold}%)")
        
        fig.update_layout(title=f"{selected_team} - {selected_component} Health", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()