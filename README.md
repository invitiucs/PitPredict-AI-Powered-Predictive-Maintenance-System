# 🏎️ PitPredict - AI-Powered Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent predictive maintenance system designed for Formula 1 and high-performance engineering applications. Uses machine learning to predict component failures before they happen by analyzing real-time sensor data.

## 🎯 Features

- 🤖 **ML-Powered Predictions**: Random Forest classifier with 95%+ accuracy
- 📊 **Real-Time Monitoring**: Live dashboard with auto-refresh capabilities
- ⚠️ **Smart Alerts**: Three-tier warning system (Healthy/Warning/Critical)
- 📈 **Historical Analysis**: Trend visualization for pattern detection
- 🎯 **Feature Engineering**: 15+ engineered features for robust predictions
- 🔄 **Simulated Sensors**: Realistic F1 component behavior simulation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/pitpredict.git
cd pitpredict

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

**Step 1: Train the Model**
```bash
python train.py
```

**Step 2: Launch Dashboard**
```bash
streamlit run dashboard.py
```

The dashboard will automatically open at `http://localhost:8501`

## 📊 Demo

### Dashboard Preview
- Real-time sensor monitoring (Vibration, Temperature, Pressure, RPM)
- AI failure probability predictions (0-100%)
- Interactive gauge charts with color-coded zones
- Historical trend analysis
- Component-specific monitoring

### Alert System
- 🟢 **Healthy (0-50%)**: Normal operation
- 🟡 **Warning (50-70%)**: Schedule maintenance
- 🔴 **Critical (70-100%)**: Immediate action required

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn (Random Forest) |
| **Visualization** | plotly, matplotlib, seaborn |
| **Dashboard** | Streamlit |
| **Model Persistence** | joblib |

## 📁 Project Structure

```
pitpredict/
├── train.py                    # Model training pipeline
├── dashboard.py                # Real-time monitoring dashboard
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Documentation
```

## 🧠 How It Works

### 1. Sensor Data Simulation
Monitors 9 key parameters:
- Vibration (mm/s)
- Temperature (°C)
- RPM (revolutions/min)
- Pressure (PSI)
- Noise Level (dB)
- Runtime Hours
- Load Cycles
- Vibration Variance
- Temperature Gradient

### 2. Feature Engineering
Creates advanced features:
- Interaction ratios (vibration/temperature, pressure/RPM)
- Polynomial features (vibration², temperature²)
- Composite risk scores

### 3. Machine Learning
- **Algorithm**: Random Forest Classifier
- **Features**: 15 engineered variables
- **Performance**: ~95% accuracy, 0.96 ROC-AUC
- **Output**: Failure probability (0-100%)

## 🎮 Usage Examples

### Normal Monitoring
```python
# Dashboard shows healthy readings
Vibration: 45 mm/s
Temperature: 75°C
Failure Probability: 15% ✅
```

### Failure Detection
```python
# Enable "Simulate Failure Mode" in sidebar
Vibration: 85 mm/s ⚠️
Temperature: 98°C ⚠️
Failure Probability: 87% 🔴
Alert: CRITICAL - Immediate maintenance required!
```

## 📈 Model Performance

```
Classification Report:
              precision    recall  f1-score
     Healthy       0.97      0.98      0.97
     Failure       0.91      0.88      0.90

ROC-AUC Score: 0.96
```

## 🔮 Future Enhancements

- [ ] Integration with real IoT sensors (Arduino/Raspberry Pi)
- [ ] LSTM networks for time-series forecasting
- [ ] Multi-component simultaneous tracking
- [ ] Cloud deployment (AWS/Azure)
- [ ] Mobile app companion
- [ ] Email/SMS alert notifications
- [ ] Database integration for historical storage

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional sensor types (oil pressure, fuel flow)
- Alternative ML algorithms (XGBoost, LightGBM)
- Enhanced visualizations (3D plots, heatmaps)
- Real F1 telemetry integration

## 📄 License

MIT License - Free for educational and commercial use

## 👨‍💻 Author

Created with ❤️ for the Formula 1 community and predictive maintenance enthusiasts

## 🙏 Acknowledgments

- Formula 1 for inspiration
- scikit-learn team for ML tools
- Streamlit for the amazing framework

---

**⭐ If you find this project useful, please give it a star!**

🏎️ Stay ahead of failure. Predict. Prevent. Win.