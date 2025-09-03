# pages/3_Smart_Maintenance_System.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import re
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available. Image processing features will be limited.")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    st.warning("⚠️ Tesseract OCR not available. Text extraction from images will be simulated.")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Configuration ---
@dataclass
class MaintenanceConfig:
    """Configuration for predictive maintenance system"""
    risk_threshold_high: float = 0.7
    risk_threshold_medium: float = 0.4
    prediction_horizon_days: int = 30
    min_confidence: float = 0.6
    image_analysis_confidence: float = 0.5

config = MaintenanceConfig()

# --- Room Schema Analysis ---
class RoomSchemaAnalyzer:
    """Analyzes room schematics and floor plans for maintenance insights"""
    
    def __init__(self):
        self.equipment_patterns = {
            'HVAC': ['AHU', 'AC', 'HVAC', 'AIR', 'VENT', 'FAN'],
            'ELECTRICAL': ['PANEL', 'SWITCH', 'OUTLET', 'UPS', 'ELECTRICAL'],
            'FIRE_SAFETY': ['FIRE', 'SMOKE', 'ALARM', 'SPRINKLER', 'EXIT'],
            'PLUMBING': ['WC', 'WASH', 'SHOWER', 'KITCHEN', 'WATER'],
            'IT_EQUIPMENT': ['SERVER', 'RACK', 'NETWORK', 'COMPUTER', 'PRINTER'],
            'FURNITURE': ['DESK', 'CHAIR', 'TABLE', 'CABINET', 'STORAGE'],
            'SECURITY': ['CAMERA', 'SENSOR', 'ACCESS', 'CARD', 'READER'],
            'AV_EQUIPMENT': ['PROJECTOR', 'SCREEN', 'SPEAKER', 'MIC', 'CAMERA']
        }
        
        self.failure_patterns = {
            'OVERCROWDING': 'High density of equipment in small area',
            'POOR_VENTILATION': 'HVAC equipment far from critical systems',
            'ELECTRICAL_OVERLOAD': 'Multiple high-power devices near single panel',
            'FIRE_RISK': 'Fire safety equipment coverage gaps',
            'ACCESS_ISSUES': 'Equipment in hard-to-reach locations',
            'WATER_DAMAGE_RISK': 'Electronic equipment near water sources'
        }

    def extract_text_from_image(self, image_path: str) -> List[str]:
        """Extract text from floor plan images using OCR or simulation"""
        try:
            if not CV2_AVAILABLE or not TESSERACT_AVAILABLE:
                # Simulate OCR results for demo
                st.info("🔄 Simulating OCR text extraction (OCR libraries not available)")
                return self._simulate_ocr_results(image_path)
            
            # Load and preprocess image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply image preprocessing for better OCR
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            
            # Clean and split text into meaningful tokens
            tokens = []
            for line in text.split('\n'):
                words = re.findall(r'[A-Za-z0-9]+', line.upper())
                tokens.extend(words)
            
            return [token for token in tokens if len(token) > 1]
            
        except Exception as e:
            st.warning(f"OCR extraction failed for {image_path}: {str(e)}")
            return self._simulate_ocr_results(image_path)

    def _simulate_ocr_results(self, image_path: str) -> List[str]:
        """Simulate OCR results for demo purposes"""
        # Generate realistic building schema text
        simulated_tokens = [
            'HVAC', 'AHU', 'AC', 'ELECTRICAL', 'PANEL', 'SWITCH', 
            'FIRE', 'ALARM', 'EXIT', 'SERVER', 'RACK', 'NETWORK',
            'DESK', 'CHAIR', 'CAMERA', 'SENSOR', 'PROJECTOR', 'SCREEN',
            'WC', 'KITCHEN', 'WATER', 'STORAGE', 'CABINET'
        ]
        
        # Randomly select tokens to simulate different schemas
        num_tokens = random.randint(10, 25)
        return random.sample(simulated_tokens, min(num_tokens, len(simulated_tokens)))

    def analyze_equipment_distribution(self, text_tokens: List[str]) -> Dict[str, Any]:
        """Analyze equipment distribution and identify potential issues"""
        equipment_found = {}
        risk_factors = []
        
        # Categorize found equipment
        for category, patterns in self.equipment_patterns.items():
            matches = [token for token in text_tokens if any(pattern in token for pattern in patterns)]
            if matches:
                equipment_found[category] = {
                    'count': len(matches),
                    'items': matches,
                    'density_score': len(matches) / len(text_tokens) if text_tokens else 0
                }
        
        # Analyze risk factors
        total_equipment = sum(data['count'] for data in equipment_found.values())
        
        # High equipment density risk
        if total_equipment > 20:  # Adjusted threshold for simulation
            risk_factors.append({
                'type': 'OVERCROWDING',
                'severity': 'HIGH',
                'description': f'High equipment density detected ({total_equipment} items)',
                'recommendation': 'Consider redistributing equipment across multiple areas'
            })
        
        # HVAC coverage analysis
        hvac_count = equipment_found.get('HVAC', {}).get('count', 0)
        it_count = equipment_found.get('IT_EQUIPMENT', {}).get('count', 0)
        
        if hvac_count < it_count * 0.1 and it_count > 0:
            risk_factors.append({
                'type': 'POOR_VENTILATION',
                'severity': 'MEDIUM',
                'description': 'Insufficient HVAC coverage for IT equipment',
                'recommendation': 'Add additional ventilation near IT equipment clusters'
            })
        
        # Fire safety analysis
        fire_safety_count = equipment_found.get('FIRE_SAFETY', {}).get('count', 0)
        if fire_safety_count < total_equipment * 0.05 and total_equipment > 0:
            risk_factors.append({
                'type': 'FIRE_RISK',
                'severity': 'HIGH',
                'description': 'Insufficient fire safety coverage',
                'recommendation': 'Install additional fire suppression systems'
            })
        
        # Water damage risk
        plumbing_count = equipment_found.get('PLUMBING', {}).get('count', 0)
        if plumbing_count > 0 and it_count > 0:
            risk_factors.append({
                'type': 'WATER_DAMAGE_RISK',
                'severity': 'MEDIUM',
                'description': 'IT equipment may be near water sources',
                'recommendation': 'Install water sensors and ensure proper drainage'
            })
        
        return {
            'equipment_found': equipment_found,
            'risk_factors': risk_factors,
            'total_equipment': total_equipment,
            'risk_score': len([r for r in risk_factors if r['severity'] == 'HIGH']) * 0.4 + 
                         len([r for r in risk_factors if r['severity'] == 'MEDIUM']) * 0.2
        }

# --- Predictive Maintenance Engine ---
class PredictiveMaintenanceEngine:
    """Advanced predictive maintenance with ML models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.equipment_profiles = {
            'HVAC': {
                'expected_life': 15 * 365,  # 15 years in days
                'maintenance_interval': 90,  # 3 months
                'failure_indicators': ['temperature_variance', 'energy_consumption', 'vibration'],
                'seasonal_factors': True
            },
            'IT_EQUIPMENT': {
                'expected_life': 5 * 365,  # 5 years
                'maintenance_interval': 180,  # 6 months
                'failure_indicators': ['cpu_temperature', 'disk_usage', 'memory_errors'],
                'seasonal_factors': False
            },
            'ELECTRICAL': {
                'expected_life': 20 * 365,  # 20 years
                'maintenance_interval': 365,  # 1 year
                'failure_indicators': ['voltage_fluctuation', 'current_load', 'temperature'],
                'seasonal_factors': False
            },
            'FIRE_SAFETY': {
                'expected_life': 10 * 365,  # 10 years
                'maintenance_interval': 180,  # 6 months
                'failure_indicators': ['sensor_sensitivity', 'battery_level', 'response_time'],
                'seasonal_factors': False
            },
            # Added for completeness, assuming AV equipment might have a profile
            'AV_EQUIPMENT': {
                'expected_life': 7 * 365, # 7 years
                'maintenance_interval': 365, # 1 year
                'failure_indicators': ['lamp_hours', 'connection_errors', 'audio_distortion'],
                'seasonal_factors': False
            }
        }

    def predict_failure_probability(self, equipment_type: str, current_metrics: Dict) -> Dict[str, Any]:
        """Predict failure probability using equipment-specific models"""
        
        if equipment_type not in self.equipment_profiles:
            # Handle unknown equipment types gracefully
            st.warning(f"No profile found for equipment type: {equipment_type}. Using default values.")
            default_profile = {
                'expected_life': 10 * 365,
                'maintenance_interval': 180
            }
            profile = default_profile
        else:
            profile = self.equipment_profiles[equipment_type]
        
        # Simulate predictive model (in real implementation, use trained models)
        base_risk = 0.1
        
        # Age factor
        age_days = current_metrics.get('age_days', 365)
        age_factor = min(age_days / profile['expected_life'], 1.0)
        
        # Usage factor
        usage_intensity = current_metrics.get('usage_intensity', 0.5)
        usage_factor = usage_intensity
        
        # Maintenance factor
        days_since_maintenance = current_metrics.get('days_since_maintenance', 30)
        maintenance_factor = min(days_since_maintenance / profile['maintenance_interval'], 1.0)
        
        # Environmental factors
        environmental_score = current_metrics.get('environmental_score', 0.5)
        
        # Calculate failure probability
        failure_probability = base_risk + (age_factor * 0.3) + (usage_factor * 0.2) + \
                            (maintenance_factor * 0.3) + (environmental_score * 0.2)
        
        failure_probability = min(failure_probability, 1.0)
        
        # Determine risk level
        if failure_probability >= config.risk_threshold_high:
            risk_level = 'HIGH'
            color = 'red'
        elif failure_probability >= config.risk_threshold_medium:
            risk_level = 'MEDIUM'
            color = 'orange'
        else:
            risk_level = 'LOW'
            color = 'green'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(equipment_type, failure_probability, current_metrics)
        
        return {
            'equipment_type': equipment_type,
            'failure_probability': failure_probability,
            'risk_level': risk_level,
            'color': color,
            'days_to_predicted_failure': int((1 - failure_probability) * profile['expected_life']),
            'recommendations': recommendations,
            'confidence': 0.85  # Simulated confidence score
        }

    def _generate_recommendations(self, equipment_type: str, failure_prob: float, metrics: Dict) -> List[str]:
        """Generate specific maintenance recommendations"""
        recommendations = []
        
        if failure_prob >= config.risk_threshold_high:
            recommendations.append(f"🚨 URGENT: Schedule immediate inspection for {equipment_type}")
            recommendations.append("📋 Create detailed maintenance plan")
            recommendations.append("💰 Budget for potential replacement")
            
        elif failure_prob >= config.risk_threshold_medium:
            recommendations.append(f"⚠️ Schedule preventive maintenance for {equipment_type}")
            recommendations.append("📊 Increase monitoring frequency")
            recommendations.append("🔍 Inspect related systems")
            
        else:
            recommendations.append(f"✅ {equipment_type} is in good condition")
            recommendations.append("📅 Continue regular maintenance schedule")
        
        # Add equipment-specific recommendations
        if equipment_type == 'HVAC' and metrics.get('energy_consumption', 0) > 0.8:
            recommendations.append("🌡️ Check and replace air filters")
            recommendations.append("🔧 Calibrate thermostats")
            
        elif equipment_type == 'IT_EQUIPMENT' and metrics.get('temperature', 0) > 0.7:
            recommendations.append("❄️ Improve cooling systems")
            recommendations.append("🧹 Clean dust from components")
            
        return recommendations

# --- Synthetic Data Generator ---
class SyntheticDataGenerator:
    """Generate synthetic maintenance data for training"""
    
    def __init__(self):
        self.equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'PLUMBING', 'AV_EQUIPMENT']
        self.failure_modes = {
            'HVAC': ['Compressor Failure', 'Filter Clog', 'Refrigerant Leak', 'Fan Motor Failure'],
            'IT_EQUIPMENT': ['Hard Drive Failure', 'CPU Overheating', 'Memory Corruption', 'Power Supply Failure'],
            'ELECTRICAL': ['Circuit Overload', 'Insulation Breakdown', 'Contact Wear', 'Voltage Fluctuation'],
            'FIRE_SAFETY': ['Sensor Degradation', 'Battery Failure', 'Alarm Malfunction', 'Sprinkler Blockage']
        }

    def generate_training_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic training data"""
        
        data = []
        
        for _ in range(num_samples):
            equipment_type = random.choice(self.equipment_types)
            
            # Generate synthetic features
            record = {
                'equipment_id': f"{equipment_type}_{random.randint(1000, 9999)}",
                'equipment_type': equipment_type,
                'age_days': random.randint(30, 3650),  # 1 month to 10 years
                'usage_intensity': random.uniform(0.1, 1.0),
                'days_since_maintenance': random.randint(1, 730),  # Up to 2 years
                'environmental_score': random.uniform(0.0, 1.0),
                'temperature_avg': random.uniform(15, 35),  # Celsius
                'humidity_avg': random.uniform(30, 80),  # Percentage
                'vibration_level': random.uniform(0, 10),
                'energy_consumption': random.uniform(0.1, 2.0),
                'error_count_30d': random.randint(0, 50),
                'maintenance_cost_total': random.uniform(500, 10000),
                'location_zone': random.choice(['A', 'B', 'C', 'D']),
                'criticality': random.choice(['LOW', 'MEDIUM', 'HIGH']),
            }
            
            # Generate failure indicators based on equipment type
            if equipment_type in self.failure_modes:
                failure_mode = random.choice(self.failure_modes[equipment_type])
                
                # Create realistic failure probability based on features
                age_factor = record['age_days'] / 2000
                usage_factor = record['usage_intensity']
                maintenance_factor = record['days_since_maintenance'] / 365
                
                failure_probability = min(0.1 + age_factor * 0.3 + usage_factor * 0.2 + 
                                        maintenance_factor * 0.4 + random.uniform(-0.1, 0.1), 1.0)
                
                record['failure_probability'] = max(failure_probability, 0.0)
                record['failure_mode'] = failure_mode
                record['days_to_failure'] = int((1 - failure_probability) * 365)
                
                # Binary failure indicator (within next 30 days)
                record['will_fail_30d'] = 1 if record['days_to_failure'] <= 30 else 0
            
            data.append(record)
        
        return pd.DataFrame(data)

    def save_synthetic_data(self, df: pd.DataFrame, filename: str = 'synthetic_maintenance_data.csv'):
        """Save generated data to CSV"""
        df.to_csv(filename, index=False)
        return filename

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="🔧 Smart Predictive Maintenance",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .risk-high { background-color: #ffebee; border-left: 5px solid #f44336; padding: 1rem; border-radius: 5px; }
        .risk-medium { background-color: #fff8e1; border-left: 5px solid #ff9800; padding: 1rem; border-radius: 5px; }
        .risk-low { background-color: #e8f5e8; border-left: 5px solid #4caf50; padding: 1rem; border-radius: 5px; }
        .metric-card { 
            background: white; 
            padding: 1.5rem; 
            border-radius: 10px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Smart Predictive Maintenance System</h1>
        <p>AI-Powered Room Schema Analysis & Failure Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    schema_analyzer = RoomSchemaAnalyzer()
    maintenance_engine = PredictiveMaintenanceEngine()
    data_generator = SyntheticDataGenerator()
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Configuration")
        
        analysis_mode = st.selectbox(
            "Select Analysis Mode",
            ["Predictive Dashboard", "Room Schema Analysis", "Equipment Monitoring", "Synthetic Data Generation"]
        )
        
        st.header("📊 System Status")
        st.metric("🏢 Monitored Rooms", "12")
        st.metric("📱 Connected Devices", "148")
        st.metric("⚠️ Active Alerts", "3")
        st.metric("📈 System Health", "92%")
    
    # Main content based on selected mode
    if analysis_mode == "Room Schema Analysis":
        handle_schema_analysis(schema_analyzer)
        
    elif analysis_mode == "Equipment Monitoring":
        handle_equipment_monitoring(maintenance_engine)
        
    elif analysis_mode == "Synthetic Data Generation":
        handle_synthetic_data_generation(data_generator)
        
    elif analysis_mode == "Predictive Dashboard":
        handle_predictive_dashboard(maintenance_engine)

def handle_schema_analysis(analyzer):
    """Handle room schema analysis interface"""
    st.subheader("🏗️ Room Schema Analysis")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Room Schematics (PNG, JPG)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.subheader(f"📄 Analysis: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Display uploaded image
                image = Image.open(temp_path)
                st.image(image, caption=f"Floor Plan: {uploaded_file.name}", use_column_width=True)
                
                # Analyze the schema
                with st.spinner("🔍 Analyzing room schema..."):
                    text_tokens = analyzer.extract_text_from_image(temp_path)
                    analysis_results = analyzer.analyze_equipment_distribution(text_tokens)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Equipment Distribution")
                    
                    equipment_found = analysis_results['equipment_found']
                    if equipment_found:
                        # Create pie chart
                        categories = list(equipment_found.keys())
                        counts = [equipment_found[cat]['count'] for cat in categories]
                        
                        fig = px.pie(
                            values=counts,
                            names=categories,
                            title="Equipment Distribution by Category"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Equipment details table
                        equipment_df = pd.DataFrame([
                            {
                                'Category': cat,
                                'Count': data['count'],
                                'Density': f"{data['density_score']:.2%}",
                                'Items': ', '.join(data['items'][:3]) + ('...' if len(data['items']) > 3 else '')
                            }
                            for cat, data in equipment_found.items()
                        ])
                        st.dataframe(equipment_df, use_container_width=True)
                    else:
                        st.warning("No equipment detected in the schema")
                
                with col2:
                    st.subheader("⚠️ Risk Assessment")
                    
                    risk_score = analysis_results['risk_score']
                    
                    # Risk score gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = risk_score * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Risk Score"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors
                st.subheader("🎯 Identified Risk Factors")
                
                risk_factors = analysis_results['risk_factors']
                if risk_factors:
                    for risk in risk_factors:
                        severity_class = f"risk-{risk['severity'].lower()}"
                        
                        st.markdown(f"""
                        <div class="{severity_class}">
                            <h4>{risk['type'].replace('_', ' ').title()}</h4>
                            <p><strong>Severity:</strong> {risk['severity']}</p>
                            <p><strong>Description:</strong> {risk['description']}</p>
                            <p><strong>Recommendation:</strong> {risk['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.success("✅ No significant risk factors identified!")
                
                # Cleanup
                os.remove(temp_path)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)

def handle_equipment_monitoring(maintenance_engine):
    """Handle equipment monitoring interface"""
    st.subheader("📱 Equipment Monitoring & Prediction")
    
    # Sample equipment data
    sample_equipment = [
        {'id': 'HVAC_001', 'type': 'HVAC', 'location': 'Main Hall', 'age_days': 1825},
        {'id': 'IT_SRV_001', 'type': 'IT_EQUIPMENT', 'location': 'Server Room', 'age_days': 912},
        {'id': 'ELEC_PAN_001', 'type': 'ELECTRICAL', 'location': 'Electrical Room', 'age_days': 2190},
        {'id': 'FIRE_001', 'type': 'FIRE_SAFETY', 'location': 'Main Entrance', 'age_days': 730}
    ]
    
    st.subheader("🏭 Equipment Status Overview")
    
    predictions = []
    
    for equipment in sample_equipment:
        # Generate sample metrics
        current_metrics = {
            'age_days': equipment['age_days'],
            'usage_intensity': random.uniform(0.3, 0.9),
            'days_since_maintenance': random.randint(10, 200),
            'environmental_score': random.uniform(0.2, 0.8),
            'temperature': random.uniform(0.3, 0.9),
            'energy_consumption': random.uniform(0.4, 1.0)
        }
        
        # Get prediction
        prediction = maintenance_engine.predict_failure_probability(
            equipment['type'], current_metrics
        )
        
        prediction['equipment_id'] = equipment['id']
        prediction['location'] = equipment['location']
        predictions.append(prediction)
    
    # Display predictions
    cols = st.columns(2)
    
    for i, pred in enumerate(predictions):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{pred['equipment_id']}</h3>
                <p><strong>Type:</strong> {pred['equipment_type']}</p>
                <p><strong>Location:</strong> {pred['location']}</p>
                <p><strong>Risk Level:</strong> <span style="color: {pred['color']}; font-weight: bold;">{pred['risk_level']}</span></p>
                <p><strong>Failure Probability:</strong> {pred['failure_probability']:.1%}</p>
                <p><strong>Predicted Days to Failure:</strong> {pred['days_to_predicted_failure']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            with st.expander("💡 View Recommendations"):
                for rec in pred['recommendations']:
                    st.write(f"• {rec}")

def handle_synthetic_data_generation(data_generator):
    """Handle synthetic data generation interface"""
    st.subheader("🔬 Synthetic Training Data Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Generate synthetic maintenance data for training machine learning models.")
        
        num_samples = st.slider("Number of samples to generate", 1000, 50000, 10000, step=1000)
        
        if st.button("🔄 Generate Synthetic Data", type="primary"):
            with st.spinner("Generating synthetic training data..."):
                synthetic_df = data_generator.generate_training_data(num_samples)
                
                # Save data
                filename = data_generator.save_synthetic_data(synthetic_df)
                
                st.success(f"✅ Generated {len(synthetic_df)} synthetic records!")
                st.info(f"📁 Data saved as: {filename}")
                
                # Display sample data
                st.subheader("📊 Sample Generated Data")
                st.dataframe(synthetic_df.head(20), use_container_width=True)
                
                # Data statistics
                st.subheader("📈 Data Statistics")
                
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.write("**Equipment Type Distribution:**")
                    type_counts = synthetic_df['equipment_type'].value_counts()
                    fig = px.bar(x=type_counts.index, y=type_counts.values, 
                               title="Equipment Types Generated")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_stats2:
                    st.write("**Failure Probability Distribution:**")
                    fig = px.histogram(synthetic_df, x='failure_probability', nbins=20,
                                     title="Failure Probability Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                csv_data = synthetic_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Synthetic Data (CSV)",
                    data=csv_data,
                    file_name=f"synthetic_maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.subheader("📋 Data Schema")
        st.write("**Generated Features:**")
        schema_info = {
            'equipment_id': 'Unique equipment identifier',
            'equipment_type': 'Type of equipment (HVAC, IT, etc.)',
            'age_days': 'Age of equipment in days',
            'usage_intensity': 'Usage intensity (0-1)',
            'days_since_maintenance': 'Days since last maintenance',
            'environmental_score': 'Environmental conditions score',
            'temperature_avg': 'Average temperature (°C)',
            'humidity_avg': 'Average humidity (%)',
            'vibration_level': 'Vibration level (0-10)',
            'energy_consumption': 'Energy consumption factor',
            'error_count_30d': 'Error count in last 30 days',
            'maintenance_cost_total': 'Total maintenance cost',
            'failure_probability': 'Predicted failure probability',
            'will_fail_30d': 'Binary: will fail in 30 days'
        }
        
        for feature, description in schema_info.items():
            st.write(f"**{feature}:** {description}")

def handle_predictive_dashboard(maintenance_engine):
    """Handle predictive maintenance dashboard"""
    st.subheader("📊 Predictive Maintenance Dashboard")
    
    # Generate sample real-time data
    np.random.seed(42)  # For consistent demo data
    
    # Time series data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Sample equipment fleet
    equipment_fleet = [
        {'id': 'HVAC_001', 'type': 'HVAC', 'location': 'Building A - Floor 1', 'criticality': 'HIGH'},
        {'id': 'HVAC_002', 'type': 'HVAC', 'location': 'Building A - Floor 2', 'criticality': 'HIGH'},
        {'id': 'IT_SRV_001', 'type': 'IT_EQUIPMENT', 'location': 'Data Center', 'criticality': 'CRITICAL'},
        {'id': 'IT_SRV_002', 'type': 'IT_EQUIPMENT', 'location': 'Data Center', 'criticality': 'CRITICAL'},
        {'id': 'ELEC_001', 'type': 'ELECTRICAL', 'location': 'Main Electrical Room', 'criticality': 'CRITICAL'},
        {'id': 'FIRE_001', 'type': 'FIRE_SAFETY', 'location': 'Building A - Lobby', 'criticality': 'HIGH'},
        {'id': 'FIRE_002', 'type': 'FIRE_SAFETY', 'location': 'Building A - Floor 3', 'criticality': 'MEDIUM'},
        {'id': 'AV_001', 'type': 'AV_EQUIPMENT', 'location': 'Conference Room A', 'criticality': 'LOW'},
    ]
    
    # Create tabs for different dashboard views
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Overview", "📈 Trends", "⚠️ Alerts", "📋 Maintenance Schedule"])
    
    with tab1:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🏭 Total Equipment",
                value=len(equipment_fleet),
                delta="2 new this month"
            )
        
        with col2:
            high_risk_count = 2  # Simulated
            st.metric(
                label="🔴 High Risk Equipment",
                value=high_risk_count,
                delta="-1 from last week",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                label="💰 Maintenance Budget Used",
                value="68%",
                delta="12% this quarter"
            )
        
        with col4:
            st.metric(
                label="⚡ System Uptime",
                value="99.2%",
                delta="0.3% improvement"
            )
        
        st.divider()
        
        # Equipment status grid
        st.subheader("🎛️ Equipment Status Grid")
        
        # Create predictions for all equipment
        all_predictions = []
        for equipment in equipment_fleet:
            current_metrics = {
                'age_days': random.randint(365, 2555),
                'usage_intensity': random.uniform(0.3, 0.9),
                'days_since_maintenance': random.randint(10, 200),
                'environmental_score': random.uniform(0.2, 0.8)
            }
            
            try:
                prediction = maintenance_engine.predict_failure_probability(
                    equipment['type'], current_metrics
                )
                
                # Ensure all required keys are present
                prediction.update({
                    'equipment_id': equipment['id'],
                    'location': equipment['location'],
                    'criticality': equipment['criticality']
                })
                
                # Add default values for any missing keys
                prediction.setdefault('failure_probability', 0.0)
                prediction.setdefault('risk_level', 'LOW')
                prediction.setdefault('days_to_predicted_failure', 365)
                prediction.setdefault('color', 'green')
                
                all_predictions.append(prediction)
                
            except Exception as e:
                st.error(f"Error processing equipment {equipment['id']}: {str(e)}")
                # Add a default prediction to prevent crashes
                default_prediction = {
                    'equipment_id': equipment['id'],
                    'location': equipment['location'],
                    'criticality': equipment['criticality'],
                    'failure_probability': 0.0,
                    'risk_level': 'LOW',
                    'days_to_predicted_failure': 365,
                    'color': 'green',
                    'equipment_type': equipment['type']
                }
                all_predictions.append(default_prediction)
        
        # Display equipment grid
        cols = st.columns(4)
        for i, pred in enumerate(all_predictions):
            with cols[i % 4]:
                # Color based on risk level
                if pred['risk_level'] == 'HIGH':
                    card_color = "#ffebee"
                    border_color = "#f44336"
                elif pred['risk_level'] == 'MEDIUM':
                    card_color = "#fff8e1"
                    border_color = "#ff9800"
                else:
                    card_color = "#e8f5e8"
                    border_color = "#4caf50"
                
                st.markdown(f"""
                <div style="
                    background-color: {card_color}; 
                    border-left: 5px solid {border_color}; 
                    padding: 1rem; 
                    border-radius: 5px; 
                    margin-bottom: 1rem;
                    height: 180px;
                ">
                    <h4 style="margin: 0; font-size: 14px;">{pred['equipment_id']}</h4>
                    <p style="margin: 5px 0; font-size: 12px; color: #666;">{pred['location']}</p>
                    <p style="margin: 5px 0; font-size: 12px;"><strong>Risk:</strong> 
                        <span style="color: {pred['color']};">{pred['risk_level']}</span>
                    </p>
                    <p style="margin: 5px 0; font-size: 12px;"><strong>Failure Prob:</strong> {pred['failure_probability']:.1%}</p>
                    <p style="margin: 5px 0; font-size: 12px;"><strong>Days to Failure:</strong> {pred['days_to_predicted_failure']}</p>
                    <p style="margin: 5px 0; font-size: 12px;"><strong>Criticality:</strong> {pred['criticality']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📈 Predictive Analytics Trends")
        
        # Generate time series data for trends
        trend_data = []
        for i, date in enumerate(dates[-90:]):  # Last 90 days
            trend_data.append({
                'date': date,
                'avg_failure_risk': 0.3 + 0.2 * np.sin(i * 0.1) + np.random.normal(0, 0.05),
                'maintenance_events': np.random.poisson(2),
                'system_health': 95 + 5 * np.sin(i * 0.05) + np.random.normal(0, 2),
                'energy_efficiency': 85 + 10 * np.sin(i * 0.03) + np.random.normal(0, 1)
            })
        
        trend_df = pd.DataFrame(trend_data)
        
        # Risk trend chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                trend_df, 
                x='date', 
                y='avg_failure_risk',
                title="Average Failure Risk Over Time",
                labels={'avg_failure_risk': 'Failure Risk', 'date': 'Date'}
            )
            fig.add_hline(y=config.risk_threshold_high, line_dash="dash", 
                          line_color="red", annotation_text="High Risk Threshold")
            fig.add_hline(y=config.risk_threshold_medium, line_dash="dash", 
                          line_color="orange", annotation_text="Medium Risk Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                trend_df,
                x='date',
                y='system_health',
                title="Overall System Health Score",
                labels={'system_health': 'Health Score (%)', 'date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance events chart
        fig = px.bar(
            trend_df,
            x='date',
            y='maintenance_events',
            title="Daily Maintenance Events",
            labels={'maintenance_events': 'Number of Events', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🚨 Active Alerts & Notifications")
        
        # Generate sample alerts
        alerts = [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'severity': 'HIGH',
                'equipment_id': 'HVAC_001',
                'alert_type': 'Predicted Failure Risk',
                'message': 'Equipment failure risk has exceeded 75%. Immediate inspection recommended.',
                'location': 'Building A - Floor 1'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=8),
                'severity': 'MEDIUM',
                'equipment_id': 'IT_SRV_002',
                'alert_type': 'Temperature Warning',
                'message': 'Server temperature consistently above normal range for 6+ hours.',
                'location': 'Data Center'
            },
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'severity': 'LOW',
                'equipment_id': 'AV_001',
                'alert_type': 'Maintenance Due',
                'message': 'Scheduled maintenance window approaching in 7 days.',
                'location': 'Conference Room A'
            }
        ]
        
        # Alert summary
        col1, col2, col3 = st.columns(3)
        with col1:
            high_alerts = len([a for a in alerts if a['severity'] == 'HIGH'])
            st.metric("🔴 High Priority", high_alerts)
        with col2:
            medium_alerts = len([a for a in alerts if a['severity'] == 'MEDIUM'])
            st.metric("🟡 Medium Priority", medium_alerts)
        with col3:
            low_alerts = len([a for a in alerts if a['severity'] == 'LOW'])
            st.metric("🟢 Low Priority", low_alerts)
        
        # Alert details
        for alert in alerts:
            # Modified severity_colors to use black text
            severity_colors = {
                'HIGH': ('#ffebee', '#f44336', 'black'),
                'MEDIUM': ('#fff8e1', '#ff9800', 'black'),
                'LOW': ('#e8f5e8', '#4caf50', 'black')
            }
            
            bg_color, border_color, text_color = severity_colors[alert['severity']]
            
            st.markdown(f"""
            <div style="
                background-color: {bg_color}; 
                border-left: 5px solid {border_color}; 
                padding: 1rem; 
                border-radius: 5px; 
                margin-bottom: 1rem;
                color: {text_color}; /* Set text color to black */
            ">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div>
                        <h4 style="margin: 0; color: {text_color};">{alert['alert_type']}</h4>
                        <p style="margin: 5px 0;"><strong>Equipment:</strong> {alert['equipment_id']} ({alert['location']})</p>
                        <p style="margin: 5px 0;">{alert['message']}</p>
                        <p style="margin: 5px 0; font-size: 12px; color: #666;">
                            {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                    </div>
                    <span style="
                        background-color: {border_color}; 
                        color: white; 
                        padding: 4px 8px; 
                        border-radius: 4px; 
                        font-size: 12px;
                        font-weight: bold;
                    ">{alert['severity']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("📋 Maintenance Schedule & Planning")
        
        # Generate sample maintenance schedule
        maintenance_schedule = []
        for i in range(20):
            equipment = random.choice(equipment_fleet)
            future_date = datetime.now() + timedelta(days=random.randint(1, 90))
            
            maintenance_schedule.append({
                'date': future_date,
                'equipment_id': equipment['id'],
                'equipment_type': equipment['type'],
                'location': equipment['location'],
                'maintenance_type': random.choice(['Preventive', 'Corrective', 'Predictive']),
                'estimated_duration': random.choice(['2 hours', '4 hours', '8 hours', '1 day']),
                'assigned_technician': random.choice(['John Smith', 'Mary Johnson', 'Bob Wilson', 'Lisa Davis']),
                'priority': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'cost_estimate': random.randint(200, 2000)
            })
        
        # Sort by date
        maintenance_schedule.sort(key=lambda x: x['date'])
        
        # Display upcoming maintenance
        st.subheader("📅 Upcoming Maintenance (Next 30 Days)")
        
        upcoming = [m for m in maintenance_schedule if m['date'] <= datetime.now() + timedelta(days=30)]
        
        if upcoming:
            # Create calendar view
            calendar_data = []
            for maintenance in upcoming:
                calendar_data.append({
                    'Date': maintenance['date'].strftime('%Y-%m-%d'),
                    'Equipment': maintenance['equipment_id'],
                    'Type': maintenance['maintenance_type'],
                    'Duration': maintenance['estimated_duration'],
                    'Technician': maintenance['assigned_technician'],
                    'Priority': maintenance['priority'],
                    'Cost': f"${maintenance['cost_estimate']}"
                })
            
            calendar_df = pd.DataFrame(calendar_data)
            
            # Color code by priority
            def highlight_priority(row):
                if row['Priority'] == 'HIGH':
                    return ['background-color: #ffebee'] * len(row)
                elif row['Priority'] == 'MEDIUM':
                    return ['background-color: #fff8e1'] * len(row)
                else:
                    return ['background-color: #e8f5e8'] * len(row)
            
            st.dataframe(
                calendar_df.style.apply(highlight_priority, axis=1),
                use_container_width=True
            )
            
            # Maintenance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_cost = sum(m['cost_estimate'] for m in upcoming)
                st.metric("💰 Total Upcoming Costs", f"${total_cost:,}")
            
            with col2:
                high_priority = len([m for m in upcoming if m['priority'] == 'HIGH'])
                st.metric("🔴 High Priority Tasks", high_priority)
            
            with col3:
                avg_cost = total_cost // len(upcoming) if upcoming else 0
                st.metric("📊 Average Task Cost", f"${avg_cost:,}")
        
        else:
            st.info("No maintenance scheduled for the next 30 days.")
        
        # Resource planning
        st.subheader("👥 Resource Allocation")
        
        technician_workload = {}
        for maintenance in upcoming:
            tech = maintenance['assigned_technician']
            if tech not in technician_workload:
                technician_workload[tech] = 0
            technician_workload[tech] += 1
        
        if technician_workload:
            workload_df = pd.DataFrame([
                {'Technician': tech, 'Assigned Tasks': count}
                for tech, count in technician_workload.items()
            ])
            
            fig = px.bar(
                workload_df,
                x='Technician',
                y='Assigned Tasks',
                title="Technician Workload Distribution (Next 30 Days)",
                color='Assigned Tasks',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Run the application ---
if __name__ == "__main__":
    main()
