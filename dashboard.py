"""
PitPredict - Real-Time Monitoring Dashboard

Save this file as: dashboard.py
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import time

# Page configuration
st.set_page_config(
    page_title="PitPredict - F1 Predictive Maintenance",
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .critical-alert {
        background-color: #ff4444;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #cc0000;
        margin: 10px 0;
    }
    .warning-alert {
        background-color: #ffbb33;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff8800;
        margin: 10px 0;
    }
    .healthy-alert {
        background-color: #00C851;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #007E33;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

class RealTimeSimulator:
    """Simulate real-time sensor data stream"""
    
    def __init__(self):
        self.base_time = datetime.now()
        self.time_offset = 0
        
    def generate_reading(self, failure_mode=False):
        """Generate single sensor reading"""
        self.time_offset += np.random.uniform(1, 3)
        
        if failure_mode:
            # Component degrading
            vibration = np.random.normal(75, 12)
            temperature = np.random.normal(95, 8)
            rpm = np.random.normal(18000, 1200)
            pressure = np.random.normal(82, 10)
            noise_level = np.random.normal(78, 6)
        else:
            # Normal operation
            vibration = np.random.normal(45, 8)
            temperature = np.random.normal(75, 6)
            rpm = np.random.normal(15000, 800)
            pressure = np.random.normal(100, 6)
            noise_level = np.random.normal(65, 5)
        
        return {
            'timestamp': self.base_time + timedelta(seconds=self.time_offset),
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

def engineer_features(df):
    """Apply same feature engineering as training"""
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

def load_model():
    """Load trained model"""
    try:
        model_data = joblib.load('pitpredict-model.pkl')
        return model_data['model'], model_data['scaler'], model_data['feature_names']
    except FileNotFoundError:
        st.error("⚠️ **Model file not found!**")
        st.info("""
        **Please train the model first by running:**
        
        ```bash
        python train.py
        ```
        
        This will create the required `pitpredict-model.pkl` file.
        
        After training completes, refresh this page.
        """)
        st.stop()
        return None, None, None  # This line won't be reached due to st.stop()
    except Exception as e:
        st.error(f"⚠️ **Error loading model:** {str(e)}")
        st.stop()
        return None, None, None

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
    st.markdown('<div class="main-header">🏎️ PitPredict</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Predictive Maintenance for F1 Components</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Control Panel")
    
    component_name = st.sidebar.selectbox(
        "Select Component",
        ["Turbocharger", "Gearbox", "Brake System", "Suspension", "Cooling System"]
    )
    
    failure_mode = st.sidebar.checkbox("🔴 Simulate Failure Mode", value=False)
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
    
    alert_threshold = st.sidebar.slider("Alert Threshold (%)", 0, 100, 50)
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.simulator = RealTimeSimulator()
    
    # Generate new reading
    reading = st.session_state.simulator.generate_reading(failure_mode)
    
    # Prepare data for prediction
    reading_df = pd.DataFrame([reading])
    reading_df = engineer_features(reading_df)
    
    # Keep only model features
    X = reading_df[feature_names]
    X_scaled = scaler.transform(X)
    
    # Predict
    failure_prob = model.predict_proba(X_scaled)[0, 1] * 100
    
    # Add to history
    st.session_state.history.append({
        'timestamp': reading['timestamp'],
        'failure_prob': failure_prob,
        'vibration': reading['vibration'],
        'temperature': reading['temperature'],
        'pressure': reading['pressure']
    })
    
    # Keep last 50 readings
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[-50:]
    
    # Status indicator
    st.markdown("---")
    
    if failure_prob >= 70:
        st.markdown(f"""
            <div class="critical-alert">
                <h2>⚠️ CRITICAL ALERT</h2>
                <h3>{component_name} - Failure Probability: {failure_prob:.1f}%</h3>
                <p>Immediate maintenance required! Component is at high risk of failure.</p>
            </div>
        """, unsafe_allow_html=True)
    elif failure_prob >= alert_threshold:
        st.markdown(f"""
            <div class="warning-alert">
                <h2>⚠️ WARNING</h2>
                <h3>{component_name} - Failure Probability: {failure_prob:.1f}%</h3>
                <p>Schedule maintenance soon. Component showing signs of degradation.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="healthy-alert">
                <h2>✅ HEALTHY</h2>
                <h3>{component_name} - Failure Probability: {failure_prob:.1f}%</h3>
                <p>Component operating within normal parameters.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Key metrics
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
        st.metric("Runtime", f"{reading['runtime_hours']:.0f} hrs",
                 delta=None)
    
    # Gauge charts
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
    st.markdown("### 📈 Historical Trends")
    
    if len(st.session_state.history) > 1:
        history_df = pd.DataFrame(st.session_state.history)
        
        # Failure probability trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
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
            height=300,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensor trends
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(history_df, x='timestamp', y=['vibration', 'temperature'],
                         title="Vibration & Temperature Trends")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(history_df, x='timestamp', y='pressure',
                         title="Pressure Trend")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Component details
    with st.expander("🔧 Component Details & Recommendations"):
        st.write(f"**Component:** {component_name}")
        st.write(f"**Last Maintenance:** {(datetime.now() - timedelta(days=int(reading['runtime_hours']/24))).strftime('%Y-%m-%d')}")
        st.write(f"**Load Cycles:** {reading['load_cycles']:,}")
        st.write(f"**Vibration Variance:** {reading['vibration_variance']:.2f}")
        
        st.markdown("**Recommendations:**")
        if failure_prob >= 70:
            st.error("🚨 Replace component immediately before next race")
            st.error("🔍 Inspect connected systems for damage")
            st.error("📋 Schedule emergency pit stop")
        elif failure_prob >= alert_threshold:
            st.warning("⚠️ Schedule preventive maintenance within 24 hours")
            st.warning("📊 Monitor closely during next session")
        else:
            st.success("✅ Continue monitoring - no action required")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()