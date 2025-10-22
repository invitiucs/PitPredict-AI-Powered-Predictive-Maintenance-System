"""
PitPredict - AI-Powered Predictive Maintenance System
Training Pipeline: Data Generation, Feature Engineering, Model Training

Save this file as: train.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class F1ComponentSimulator:
    """Simulate sensor data for F1 components with realistic failure patterns"""
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def generate_data(self):
        """Generate realistic sensor data with failure patterns"""
        data = []
        
        for i in range(self.n_samples):
            # Base parameters
            runtime_hours = np.random.uniform(0, 500)
            
            # Determine if component will fail (20% failure rate)
            will_fail = np.random.random() < 0.2
            
            if will_fail:
                # Components near failure show elevated readings
                vibration = np.random.normal(75, 15)  # Higher vibration
                temperature = np.random.normal(95, 10)  # Higher temp
                rpm = np.random.normal(18000, 1500)
                pressure = np.random.normal(85, 15)  # Lower pressure
                noise_level = np.random.normal(78, 8)
                
                # Add correlation with runtime
                if runtime_hours > 300:
                    vibration += (runtime_hours - 300) * 0.1
                    temperature += (runtime_hours - 300) * 0.05
                    
            else:
                # Healthy components
                vibration = np.random.normal(45, 10)
                temperature = np.random.normal(75, 8)
                rpm = np.random.normal(15000, 1000)
                pressure = np.random.normal(100, 8)
                noise_level = np.random.normal(65, 6)
            
            # Additional features
            load_cycles = int(runtime_hours * np.random.uniform(50, 100))
            vibration_variance = np.abs(np.random.normal(2, 1)) if not will_fail else np.abs(np.random.normal(8, 3))
            temp_gradient = np.random.uniform(-2, 2) if not will_fail else np.random.uniform(0, 5)
            
            data.append({
                'runtime_hours': runtime_hours,
                'vibration': max(0, vibration),
                'temperature': max(20, temperature),
                'rpm': max(0, rpm),
                'pressure': max(0, pressure),
                'noise_level': max(30, noise_level),
                'load_cycles': load_cycles,
                'vibration_variance': vibration_variance,
                'temp_gradient': temp_gradient,
                'failure': 1 if will_fail else 0
            })
        
        return pd.DataFrame(data)

class PitPredictTrainer:
    """Train and evaluate predictive maintenance model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def engineer_features(self, df):
        """Create advanced features for better predictions"""
        df = df.copy()
        
        # Interaction features
        df['vibration_temp_ratio'] = df['vibration'] / (df['temperature'] + 1)
        df['pressure_rpm_ratio'] = df['pressure'] / (df['rpm'] + 1)
        df['runtime_load_ratio'] = df['runtime_hours'] / (df['load_cycles'] + 1)
        
        # Polynomial features for critical sensors
        df['vibration_squared'] = df['vibration'] ** 2
        df['temperature_squared'] = df['temperature'] ** 2
        
        # Risk score (manual feature)
        df['risk_score'] = (
            (df['vibration'] / 100) * 0.3 +
            (df['temperature'] / 120) * 0.25 +
            (df['runtime_hours'] / 500) * 0.2 +
            ((100 - df['pressure']) / 100) * 0.15 +
            (df['vibration_variance'] / 15) * 0.1
        )
        
        return df
    
    def train(self, X_train, y_train):
        """Train Random Forest model"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X_train.columns.tolist()
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        print("=" * 60)
        print("PITPREDICT MODEL EVALUATION")
        print("=" * 60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Healthy', 'Failure']))
        
        print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        return y_pred, y_proba
    
    def plot_results(self, X_test, y_test, y_proba):
        """Visualize model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, self.model.predict(self.scaler.transform(X_test)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Feature Importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        axes[1, 0].barh(range(10), importances[indices], color='skyblue')
        axes[1, 0].set_yticks(range(10))
        axes[1, 0].set_yticklabels([self.feature_names[i] for i in indices])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 0].invert_yaxis()
        
        # 4. Prediction Distribution
        axes[1, 1].hist(y_proba[y_test == 0], bins=30, alpha=0.6, label='Healthy', color='green')
        axes[1, 1].hist(y_proba[y_test == 1], bins=30, alpha=0.6, label='Failure', color='red')
        axes[1, 1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        axes[1, 1].set_xlabel('Failure Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('pitpredict_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✅ Evaluation plots saved as 'pitpredict_evaluation.png'")
        plt.show()
    
    def save_model(self, filename='pitpredict_model.pkl'):
        """Save trained model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, filename)
        print(f"\n✅ Model saved as '{filename}'")

def main():
    """Main training pipeline"""
    print("🏎️  PITPREDICT - Training Pipeline")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n📊 Generating synthetic F1 sensor data...")
    simulator = F1ComponentSimulator(n_samples=10000)
    df = simulator.generate_data()
    print(f"Generated {len(df)} samples")
    print(f"Failure rate: {df['failure'].mean()*100:.1f}%")
    
    # 2. Feature engineering
    print("\n🔧 Engineering features...")
    trainer = PitPredictTrainer()
    df_engineered = trainer.engineer_features(df)
    
    # 3. Prepare data
    X = df_engineered.drop('failure', axis=1)
    y = df_engineered['failure']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # 4. Train model
    print("\n🤖 Training Random Forest model...")
    trainer.train(X_train, y_train)
    
    # 5. Evaluate
    print("\n📈 Evaluating model...")
    y_pred, y_proba = trainer.evaluate(X_test, y_test)
    
    # 6. Visualize
    print("\n📊 Creating visualizations...")
    trainer.plot_results(X_test, y_test, y_proba)
    
    # 7. Save model
    trainer.save_model()
    
    # 8. Demo prediction
    print("\n" + "=" * 60)
    print("🎯 DEMO PREDICTION")
    print("=" * 60)
    
    # Simulate a component near failure
    demo_data = pd.DataFrame([{
        'runtime_hours': 380,
        'vibration': 82,
        'temperature': 98,
        'rpm': 17500,
        'pressure': 78,
        'noise_level': 81,
        'load_cycles': 28000,
        'vibration_variance': 9.5,
        'temp_gradient': 4.2
    }])
    
    demo_data = trainer.engineer_features(demo_data)
    demo_scaled = trainer.scaler.transform(demo_data)
    
    failure_prob = trainer.model.predict_proba(demo_scaled)[0, 1]
    
    print("\nComponent Status:")
    print(f"  Runtime: 380 hours")
    print(f"  Vibration: 82 mm/s")
    print(f"  Temperature: 98°C")
    print(f"  Failure Probability: {failure_prob*100:.1f}%")
    
    if failure_prob > 0.7:
        print("\n  ⚠️  CRITICAL - Immediate maintenance required!")
    elif failure_prob > 0.5:
        print("\n  ⚠️  WARNING - Schedule maintenance soon")
    else:
        print("\n  ✅ HEALTHY - Component operating normally")
    
    print("\n" + "=" * 60)
    print("✅ Training pipeline complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()