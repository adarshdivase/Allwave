# pages/3_Smart_Maintenance_System.py
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
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
        """Extract text from floor plan images using OCR"""
        try:
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
            return []

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
        if total_equipment > 50:
            risk_factors.append({
                'type': 'OVERCROWDING',
                'severity': 'HIGH',
                'description': f'High equipment density detected ({total_equipment} items)',
                'recommendation': 'Consider redistributing equipment across multiple areas'
            })
        
        # HVAC coverage analysis
        hvac_count = equipment_found.get('HVAC', {}).get('count', 0)
        it_count = equipment_found.get('IT_EQUIPMENT', {}).get('count', 0)
        
        if hvac_count < it_count * 0.1:
            risk_factors.append({
                'type': 'POOR_VENTILATION',
                'severity': 'MEDIUM',
                'description': 'Insufficient HVAC coverage for IT equipment',
                'recommendation': 'Add additional ventilation near IT equipment clusters'
            })
        
        # Fire safety analysis
        fire_safety_count = equipment_found.get('FIRE_SAFETY', {}).get('count', 0)
        if fire_safety_count < total_equipment * 0.05:
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
            }
        }

    def predict_failure_probability(self, equipment_type: str, current_metrics: Dict) -> Dict[str, Any]:
        """Predict failure probability using equipment-specific models"""
        
        if equipment_type not in self.equipment_profiles:
            return {'error': f'Unknown equipment type: {equipment_type}'}
        
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
            ["Room Schema Analysis", "Equipment Monitoring", "Synthetic Data Generation", "Predictive Dashboard"]
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
        "Upload Room Schematics (PDF, PNG, JPG)",
        type=['pdf', 'png', 'jpg', 'jpeg'],
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
                # Convert PDF to image if needed
                if uploaded_file.name.endswith('.pdf'):
                    st.warning("PDF processing requires additional setup. Please upload image files for demo.")
                    continue
                
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
                        mode = "gauge+number+delta",
                        value = risk_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Overall Risk Score"},
                        delta = {'reference': 0.5},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.4], 'color': "lightgreen"},
                                {'range': [0.4, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7
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
            st.subheader("💡 Recommendations")
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
                    fig = px.histogram(synthetic_df, x='failure_probability', bins=20,
                                       title="Failure Probability Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                csv = synthetic_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Synthetic Data",
                    data=csv,
                    file_name=f"synthetic_maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.subheader("📋 Data Features")
        st.write("""
        **Generated features include:**
        
        • Equipment ID & Type
        • Age and usage metrics
        • Environmental conditions
        • Maintenance history
        • Failure indicators
        • Location and criticality
        • Energy consumption
        • Error counts
        • Predicted failure modes
        """)

def handle_predictive_dashboard(maintenance_engine):
    """Handle predictive maintenance dashboard"""
    st.subheader("📊 Predictive Maintenance Dashboard")
    
    # Generate sample dashboard data
    np.random.seed(42)  # For consistent demo data
    
    # Create sample equipment fleet
    equipment_fleet = []
    equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY']
    
    for i in range(50):
        equipment_type = np.random.choice(equipment_types)
        equipment_fleet.append({
            'id': f"{equipment_type}_{i+1:03d}",
            'type': equipment_type,
            'age_days': np.random.randint(30, 2000),
            'usage_intensity': np.random.uniform(0.2, 0.9),
            'days_since_maintenance': np.random.randint(1, 300),
            'environmental_score': np.random.uniform(0.1, 0.9)
        })
    
    # Generate predictions
    all_predictions = []
    for equipment in equipment_fleet:
        current_metrics = {
            'age_days': equipment['age_days'],
            'usage_intensity': equipment['usage_intensity'],
            'days_since_maintenance': equipment['days_since_maintenance'],
            'environmental_score': equipment['environmental_score'],
            'temperature': np.random.uniform(0.2, 0.9),
            'energy_consumption': np.random.uniform(0.3, 1.0)
        }
        
        prediction = maintenance_engine.predict_failure_probability(
            equipment['type'], current_metrics
        )
        
        prediction['equipment_id'] = equipment['id']
        prediction['age_days'] = equipment['age_days']
        prediction['usage_intensity'] = equipment['usage_intensity']
        all_predictions.append(prediction)
    
    # Create dashboard metrics
    high_risk_count = len([p for p in all_predictions if p['risk_level'] == 'HIGH'])
    medium_risk_count = len([p for p in all_predictions if p['risk_level'] == 'MEDIUM'])
    low_risk_count = len([p for p in all_predictions if p['risk_level'] == 'LOW'])
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔴 High Risk Equipment",
            value=high_risk_count,
            delta=f"{high_risk_count/len(all_predictions)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="🟡 Medium Risk Equipment",
            value=medium_risk_count,
            delta=f"{medium_risk_count/len(all_predictions)*100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="🟢 Low Risk Equipment",
            value=low_risk_count,
            delta=f"{low_risk_count/len(all_predictions)*100:.1f}%"
        )
    
    with col4:
        avg_failure_prob = np.mean([p['failure_probability'] for p in all_predictions])
        st.metric(
            label="📊 Avg Failure Probability",
            value=f"{avg_failure_prob:.1%}",
            delta=f"{'↑' if avg_failure_prob > 0.5 else '↓'} Risk"
        )
    
    # Charts and visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📈 Risk Distribution by Equipment Type")
        
        # Create risk distribution chart
        risk_data = []
        for equipment_type in equipment_types:
            type_predictions = [p for p in all_predictions if p['equipment_type'] == equipment_type]
            
            for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
                count = len([p for p in type_predictions if p['risk_level'] == risk_level])
                risk_data.append({
                    'Equipment Type': equipment_type,
                    'Risk Level': risk_level,
                    'Count': count
                })
        
        risk_df = pd.DataFrame(risk_data)
        
        fig = px.bar(
            risk_df,
            x='Equipment Type',
            y='Count',
            color='Risk Level',
            color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'},
            title="Risk Distribution by Equipment Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.subheader("🎯 Failure Probability vs Equipment Age")
        
        # Scatter plot of age vs failure probability
        plot_data = pd.DataFrame([
            {
                'Equipment ID': p['equipment_id'],
                'Age (Days)': p['age_days'],
                'Failure Probability': p['failure_probability'],
                'Equipment Type': p['equipment_type'],
                'Risk Level': p['risk_level']
            }
            for p in all_predictions
        ])
        
        fig = px.scatter(
            plot_data,
            x='Age (Days)',
            y='Failure Probability',
            color='Equipment Type',
            size='Failure Probability',
            hover_data=['Equipment ID', 'Risk Level'],
            title="Equipment Age vs Failure Risk"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # High-priority equipment table
    st.subheader("🚨 Priority Maintenance Schedule")
    
    # Filter high and medium risk equipment
    priority_equipment = [p for p in all_predictions if p['risk_level'] in ['HIGH', 'MEDIUM']]
    priority_equipment.sort(key=lambda x: x['failure_probability'], reverse=True)
    
    if priority_equipment:
        priority_df = pd.DataFrame([
            {
                'Equipment ID': p['equipment_id'],
                'Type': p['equipment_type'],
                'Risk Level': p['risk_level'],
                'Failure Probability': f"{p['failure_probability']:.1%}",
                'Days to Predicted Failure': p['days_to_predicted_failure'],
                'Primary Recommendation': p['recommendations'][0] if p['recommendations'] else 'N/A'
            }
            for p in priority_equipment[:10]  # Show top 10
        ])
        
        # Style the dataframe
        def color_risk_level(val):
            if val == 'HIGH':
                return 'background-color: #ffcdd2'
            elif val == 'MEDIUM':
                return 'background-color: #fff3e0'
            return ''
        
        styled_df = priority_df.style.applymap(color_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Maintenance timeline
        st.subheader("📅 Maintenance Timeline")
        
        # Create Gantt-like chart for maintenance schedule
        timeline_data = []
        base_date = datetime.now()
        
        for i, p in enumerate(priority_equipment[:10]):
            start_date = base_date + timedelta(days=i*2)  # Stagger maintenance
            end_date = start_date + timedelta(days=1)
            
            timeline_data.append({
                'Task': f"Maintain {p['equipment_id']}",
                'Start': start_date,
                'Finish': end_date,
                'Risk': p['risk_level'],
                'Equipment': p['equipment_type']
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig = px.timeline(
            timeline_df,
            x_start="Start",
            x_end="Finish",
            y="Task",
            color="Risk",
            color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange'},
            title="Scheduled Maintenance Timeline"
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.success("🎉 No high-priority maintenance required at this time!")
    
    # Advanced analytics section
    with st.expander("🔬 Advanced Analytics"):
        st.subheader("📊 Detailed Equipment Analysis")
        
        # Equipment selection for detailed analysis
        selected_equipment = st.selectbox(
            "Select Equipment for Detailed Analysis",
            options=[p['equipment_id'] for p in all_predictions]
        )
        
        if selected_equipment:
            equipment_pred = next(p for p in all_predictions if p['equipment_id'] == selected_equipment)
            
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.subheader(f"📱 {selected_equipment} Details")
                
                # Equipment details
                st.write(f"**Equipment Type:** {equipment_pred['equipment_type']}")
                st.write(f"**Risk Level:** {equipment_pred['risk_level']}")
                st.write(f"**Failure Probability:** {equipment_pred['failure_probability']:.1%}")
                st.write(f"**Predicted Days to Failure:** {equipment_pred['days_to_predicted_failure']}")
                st.write(f"**Confidence:** {equipment_pred['confidence']:.1%}")
                
                # Risk gauge for individual equipment
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = equipment_pred['failure_probability'],
                    title = {'text': f"{selected_equipment} Risk Level"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 1]},
                        'bar': {'color': equipment_pred['color']},
                        'steps': [
                            {'range': [0, 0.4], 'color': "lightgreen"},
                            {'range': [0.4, 0.7], 'color': "yellow"},
                            {'range': [0.7, 1], 'color': "red"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col_detail2:
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(equipment_pred['recommendations'], 1):
                    st.write(f"{i}. {rec}")
                
                # Historical trend simulation (placeholder)
                st.subheader("📈 Risk Trend (Simulated)")
                
                # Generate simulated historical data
                dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
                base_risk = equipment_pred['failure_probability']
                trend_data = []
                
                for i, date in enumerate(dates):
                    # Simulate gradual increase in risk over time
                    risk = max(0, base_risk - 0.3 + (i/len(dates)) * 0.3 + np.random.normal(0, 0.02))
                    trend_data.append({'Date': date, 'Risk': min(risk, 1.0)})
                
                trend_df = pd.DataFrame(trend_data)
                
                fig = px.line(
                    trend_df,
                    x='Date',
                    y='Risk',
                    title=f'{selected_equipment} Risk Trend (30 Days)',
                    labels={'Risk': 'Failure Probability'}
                )
                fig.add_hline(y=config.risk_threshold_high, line_dash="dash", line_color="red",
                              annotation_text="High Risk Threshold")
                fig.add_hline(y=config.risk_threshold_medium, line_dash="dash", line_color="orange",
                              annotation_text="Medium Risk Threshold")
                
                st.plotly_chart(fig, use_container_width=True)

# Run the application
if __name__ == "__main__":
    main()
