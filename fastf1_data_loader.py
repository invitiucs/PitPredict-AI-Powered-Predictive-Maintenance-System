"""
FastF1 Integration for PitPredict
Fetch real Formula 1 telemetry data for analysis
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"✅ Created cache directory: {cache_dir}")

# Enable FastF1 cache for faster loading
fastf1.Cache.enable_cache(cache_dir)

class FastF1DataLoader:
    """Load and process real F1 telemetry data using FastF1 API"""
    
    def __init__(self):
        self.session = None
        self.laps = None
        
    def load_session(self, year, grand_prix, session_type='R'):
        """
        Load F1 session data
        
        Parameters:
        - year: int (e.g., 2024, 2023)
        - grand_prix: str (e.g., 'Monaco', 'Bahrain', 'Abu Dhabi')
        - session_type: str
            'FP1' - Free Practice 1
            'FP2' - Free Practice 2
            'FP3' - Free Practice 3
            'Q' - Qualifying
            'R' - Race
        """
        try:
            print(f"Loading {year} {grand_prix} {session_type}...")
            self.session = fastf1.get_session(year, grand_prix, session_type)
            self.session.load()
            print(f"✅ Session loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading session: {e}")
            return False
    
    def get_driver_telemetry(self, driver_code, lap_number=None):
        """
        Get telemetry data for a specific driver
        
        Parameters:
        - driver_code: str (e.g., 'VER', 'HAM', 'LEC')
        - lap_number: int (optional, get specific lap, otherwise fastest lap)
        """
        if not self.session:
            print("⚠️ No session loaded. Call load_session() first.")
            return None
        
        try:
            driver_laps = self.session.laps.pick_driver(driver_code)
            
            if lap_number:
                lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
            else:
                # Get fastest lap
                lap = driver_laps.pick_fastest()
            
            telemetry = lap.get_telemetry()
            
            print(f"✅ Retrieved telemetry for {driver_code}")
            print(f"   Lap: {lap['LapNumber']}")
            print(f"   Lap Time: {lap['LapTime']}")
            print(f"   Data points: {len(telemetry)}")
            
            return telemetry
            
        except Exception as e:
            print(f"❌ Error getting telemetry: {e}")
            return None
    
    def convert_to_pitpredict_format(self, telemetry_df):
        """
        Convert FastF1 telemetry to PitPredict sensor format
        
        FastF1 provides:
        - Speed (km/h)
        - RPM
        - Throttle (0-100%)
        - Brake (0-100%)
        - nGear
        - DRS
        
        We convert to PitPredict format:
        - vibration, temperature, pressure, rpm, noise_level, etc.
        """
        if telemetry_df is None or len(telemetry_df) == 0:
            return None
        
        # Sample every 10th point to reduce data size
        sampled = telemetry_df.iloc[::10].copy()
        
        pitpredict_data = []
        
        for idx, row in sampled.iterrows():
            # Convert F1 telemetry to maintenance sensors
            # These are approximations based on racing conditions
            
            rpm = row['RPM'] if 'RPM' in row and pd.notna(row['RPM']) else 15000
            speed = row['Speed'] if 'Speed' in row and pd.notna(row['Speed']) else 200
            throttle = row['Throttle'] if 'Throttle' in row and pd.notna(row['Throttle']) else 50
            
            # Handle brake - can be boolean or numeric
            if 'Brake' in row and pd.notna(row['Brake']):
                brake = 100 if row['Brake'] is True else (0 if row['Brake'] is False else float(row['Brake']))
            else:
                brake = 0
            
            # Estimate sensor values based on driving conditions
            # High RPM + High speed = more vibration and heat
            vibration = 30 + (rpm / 300) + (speed / 10)
            temperature = 60 + (rpm / 400) + (throttle / 2)
            pressure = 100 - (brake / 2)  # Brake pressure inverse
            noise_level = 50 + (rpm / 400)
            
            # Add realistic noise
            vibration += np.random.normal(0, 3)
            temperature += np.random.normal(0, 2)
            pressure += np.random.normal(0, 5)
            
            pitpredict_data.append({
                'timestamp': row['Time'] if 'Time' in row else idx,
                'runtime_hours': idx * 0.001,  # Simulated
                'vibration': max(0, vibration),
                'temperature': max(20, temperature),
                'rpm': rpm,
                'pressure': max(0, pressure),
                'noise_level': max(30, noise_level),
                'load_cycles': int(idx * 10),
                'vibration_variance': np.random.uniform(2, 8),
                'temp_gradient': np.random.uniform(-2, 5),
                # Original F1 data for reference
                'f1_speed': speed,
                'f1_throttle': throttle,
                'f1_brake': brake
            })
        
        return pd.DataFrame(pitpredict_data)
    
    def get_race_statistics(self):
        """Get overall race statistics"""
        if not self.session:
            return None
        
        try:
            results = self.session.results
            
            print("\n" + "="*60)
            print(f"RACE STATISTICS: {self.session.event['EventName']}")
            print("="*60)
            
            for idx, driver in results.iterrows():
                print(f"{driver['Position']:2.0f}. {driver['Abbreviation']:3s} - "
                      f"{driver['TeamName']:25s} - {driver['Status']}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return None
    
    def analyze_component_stress(self, driver_code):
        """
        Analyze component stress throughout a race
        Returns potential failure indicators
        """
        telemetry = self.get_driver_telemetry(driver_code)
        
        if telemetry is None:
            return None
        
        analysis = {
            'driver': driver_code,
            'max_rpm': telemetry['RPM'].max(),
            'avg_rpm': telemetry['RPM'].mean(),
            'max_speed': telemetry['Speed'].max(),
            'avg_speed': telemetry['Speed'].mean(),
            'full_throttle_pct': (telemetry['Throttle'] > 95).sum() / len(telemetry) * 100,
            'heavy_braking_pct': (telemetry['Brake'] > 80).sum() / len(telemetry) * 100,
        }
        
        # Estimate failure risk based on stress
        risk_score = 0
        
        if analysis['max_rpm'] > 18000:
            risk_score += 20
        if analysis['full_throttle_pct'] > 60:
            risk_score += 15
        if analysis['heavy_braking_pct'] > 20:
            risk_score += 15
        
        analysis['estimated_failure_risk'] = min(risk_score, 100)
        
        print("\n" + "="*60)
        print(f"COMPONENT STRESS ANALYSIS: {driver_code}")
        print("="*60)
        print(f"Max RPM: {analysis['max_rpm']:.0f}")
        print(f"Avg RPM: {analysis['avg_rpm']:.0f}")
        print(f"Max Speed: {analysis['max_speed']:.1f} km/h")
        print(f"Full Throttle: {analysis['full_throttle_pct']:.1f}%")
        print(f"Heavy Braking: {analysis['heavy_braking_pct']:.1f}%")
        print(f"\n⚠️ Estimated Failure Risk: {analysis['estimated_failure_risk']:.0f}%")
        print("="*60)
        
        return analysis


def demo_fastf1_integration():
    """Demonstration of FastF1 integration"""
    
    print("="*60)
    print("PITPREDICT + FASTF1 INTEGRATION DEMO")
    print("="*60)
    
    # Initialize loader
    loader = FastF1DataLoader()
    
    # Example 1: Load recent race
    print("\n📥 Loading 2024 Monaco Grand Prix Race...")
    success = loader.load_session(2024, 'Monaco', 'R')
    
    if not success:
        print("\n⚠️ Using 2023 data as fallback...")
        loader.load_session(2023, 'Monaco', 'R')
    
    # Example 2: Get race statistics
    print("\n📊 Race Results:")
    loader.get_race_statistics()
    
    # Example 3: Get driver telemetry
    print("\n🏎️ Getting Max Verstappen's telemetry...")
    telemetry = loader.get_driver_telemetry('VER')
    
    if telemetry is not None:
        # Example 4: Convert to PitPredict format
        print("\n🔧 Converting to PitPredict sensor format...")
        pitpredict_data = loader.convert_to_pitpredict_format(telemetry)
        
        if pitpredict_data is not None:
            print(f"✅ Converted {len(pitpredict_data)} data points")
            print("\nSample PitPredict Data:")
            print(pitpredict_data.head())
            
            # Save for use in dashboard
            pitpredict_data.to_csv('f1_telemetry_pitpredict.csv', index=False)
            print("\n✅ Saved to 'f1_telemetry_pitpredict.csv'")
    
    # Example 5: Component stress analysis
    print("\n🔍 Analyzing component stress...")
    loader.analyze_component_stress('VER')
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    demo_fastf1_integration()