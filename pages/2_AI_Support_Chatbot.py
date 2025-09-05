# app.py
import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Generator
import fitz  # PyMuPDF
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
import google.generativeai as genai
import mailbox
from email import policy
from email.parser import BytesParser
from PIL import Image
import io
import time
import json

# --- Basic Configuration ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Advanced Equipment Diagnostic AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced RAG Configuration ---
@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.4

config = RAGConfig()

# --- Advanced Equipment Database with Detailed Diagnostics ---
EQUIPMENT_DATABASE = {
    'hvac': {
        'name': 'HVAC System',
        'components': ['compressor', 'condenser', 'evaporator', 'expansion valve', 'refrigerant lines', 'thermostat', 'blower motor'],
        'sensors': ['temperature', 'pressure', 'current', 'voltage', 'airflow'],
        'detailed_issues': {
            'no cooling': {
                'symptoms': ['warm air from vents', 'system running but no cooling', 'high energy consumption'],
                'diagnostic_flow': [
                    {
                        'step': 1,
                        'action': 'Check thermostat settings',
                        'instruction': 'Verify thermostat is set to COOL mode and temperature is below room temp',
                        'expected_result': 'Thermostat shows COOL mode, set temperature < room temperature',
                        'troubleshoot': 'If incorrect, adjust settings. If correct, proceed to next step.',
                        'sensors': ['temperature'],
                        'normal_values': {'temperature': '68-75°F'}
                    },
                    {
                        'step': 2,
                        'action': 'Inspect air filter',
                        'instruction': 'Remove and examine air filter for dirt, debris, or damage',
                        'expected_result': 'Clean air filter with minimal restriction',
                        'troubleshoot': 'Replace if dirty/damaged. A clogged filter reduces airflow by 40-60%.',
                        'sensors': ['airflow'],
                        'normal_values': {'airflow': '400-600 CFM'}
                    },
                    {
                        'step': 3,
                        'action': 'Check refrigerant pressure',
                        'instruction': 'Connect gauges to high and low pressure ports. System should be running.',
                        'expected_result': 'Low side: 40-50 PSI, High side: 200-250 PSI (R-410A)',
                        'troubleshoot': 'Low pressure indicates leak. High pressure suggests overcharge or restriction.',
                        'sensors': ['pressure'],
                        'normal_values': {'pressure': 'Low: 40-50 PSI, High: 200-250 PSI'}
                    },
                    {
                        'step': 4,
                        'action': 'Test compressor amperage',
                        'instruction': 'Use clamp meter to measure compressor current draw',
                        'expected_result': 'Current within 10% of nameplate rating',
                        'troubleshoot': 'High current: overcharge/restriction. Low current: undercharge/failing compressor.',
                        'sensors': ['current'],
                        'normal_values': {'current': '15-25 Amps (typical 3-ton unit)'}
                    },
                    {
                        'step': 5,
                        'action': 'Inspect condenser coils',
                        'instruction': 'Visually check outdoor coil for dirt, debris, or damage',
                        'expected_result': 'Clean coils with clear airflow path',
                        'troubleshoot': 'Clean with coil cleaner and water. Dirty coils increase head pressure by 30%.',
                        'sensors': ['temperature', 'pressure'],
                        'normal_values': {'temperature': 'Ambient + 15-25°F'}
                    }
                ]
            },
            'poor airflow': {
                'symptoms': ['weak air from vents', 'uneven cooling', 'increased noise'],
                'diagnostic_flow': [
                    {
                        'step': 1,
                        'action': 'Measure supply air temperature',
                        'instruction': 'Insert thermometer into supply vent closest to unit',
                        'expected_result': '14-20°F below return air temperature',
                        'troubleshoot': 'If temperature difference is less than 14°F, airflow issue confirmed.',
                        'sensors': ['temperature'],
                        'normal_values': {'temperature': 'ΔT = 14-20°F'}
                    },
                    {
                        'step': 2,
                        'action': 'Check blower motor operation',
                        'instruction': 'Listen for unusual noises, vibration, or intermittent operation',
                        'expected_result': 'Smooth, quiet operation at all speeds',
                        'troubleshoot': 'Replace bearings if noisy. Check capacitor if speed issues.',
                        'sensors': ['current', 'voltage'],
                        'normal_values': {'current': '8-12 Amps', 'voltage': '208-240V'}
                    }
                ]
            }
        }
    },
    'electrical': {
        'name': 'Electrical Panel',
        'components': ['main breaker', 'branch breakers', 'bus bars', 'neutral bar', 'ground bar', 'meter'],
        'sensors': ['voltage', 'current', 'power', 'temperature', 'resistance'],
        'detailed_issues': {
            'power outage': {
                'symptoms': ['complete loss of power', 'partial circuits out', 'flickering lights'],
                'diagnostic_flow': [
                    {
                        'step': 1,
                        'action': 'Check main breaker status',
                        'instruction': 'Visually inspect main breaker position and look for trip indicators',
                        'expected_result': 'Main breaker in ON position, no trip indicators visible',
                        'troubleshoot': 'If tripped, determine cause before resetting. Check for overload.',
                        'sensors': ['voltage'],
                        'normal_values': {'voltage': '230-240V across main'}
                    },
                    {
                        'step': 2,
                        'action': 'Test incoming power',
                        'instruction': 'Use multimeter to check voltage at main breaker input terminals',
                        'expected_result': '230-240V between hot legs, 115-120V to neutral',
                        'troubleshoot': 'If no voltage, contact utility company. If present, check connections.',
                        'sensors': ['voltage'],
                        'normal_values': {'voltage': 'L1-L2: 240V, L1-N: 120V, L2-N: 120V'}
                    }
                ]
            }
        }
    },
    'network': {
        'name': 'Network Switch',
        'components': ['ports', 'power supply', 'cooling fan', 'status LEDs', 'management interface'],
        'sensors': ['temperature', 'power', 'traffic', 'errors'],
        'detailed_issues': {
            'connection timeout': {
                'symptoms': ['devices cant connect', 'intermittent connectivity', 'slow response times'],
                'diagnostic_flow': [
                    {
                        'step': 1,
                        'action': 'Check cable integrity',
                        'instruction': 'Use cable tester to verify continuity on suspected cable',
                        'expected_result': 'All 8 wires show continuity, no shorts or opens',
                        'troubleshoot': 'Replace cable if any wire fails. Check connectors for damage.',
                        'sensors': ['resistance'],
                        'normal_values': {'resistance': '<100 ohms per wire'}
                    }
                ]
            }
        }
    }
}

# --- Enhanced Maintenance Pipeline ---
class MaintenancePipeline:
    def __init__(self):
        self.maintenance_data = self._load_maintenance_data()
        self.equipment_list = list(self.maintenance_data.keys())

    def _load_maintenance_data(self) -> Dict:
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'Network Switch', 'Server', 'Crestron Panel']
        for i in range(30):
            eq_type = random.choice(equipment_types)
            fail_prob = random.uniform(0.1, 0.9)
            equipment_data[f"{eq_type.replace(' ','_')}_{i+1}"] = {
                'type': eq_type,
                'location': f"Building A - Floor {random.randint(1,4)} - Rack {chr(65+i%4)}",
                'failure_probability': fail_prob,
                'risk_level': 'HIGH' if fail_prob > 0.7 else 'MEDIUM' if fail_prob > 0.4 else 'LOW',
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 5000)
            }
        return equipment_data

    def get_equipment_by_risk(self, risk_level: str) -> List[Dict]:
        return [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() if edata['risk_level'] == risk_level.upper()]

    def simulate_real_time_alert(self) -> Dict:
        alert_type = random.choice(["Device Offline", "High Temperature", "High CPU Usage", "Configuration Mismatch"])
        device = random.choice(self.equipment_list)
        device_info = self.maintenance_data[device]
        return {
            "alert_type": alert_type,
            "device_id": device,
            "device_type": device_info['type'],
            "location": device_info['location'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# --- Sensor Data Simulation ---
def generate_sensor_data(sensor_type: str, equipment_type: str = None) -> str:
    """Generate realistic sensor data based on type"""
    if sensor_type == 'temperature':
        return f"{(72 + random.uniform(-5, 15)):.1f}°F"
    elif sensor_type == 'pressure':
        low = 45 + random.uniform(-10, 10)
        high = 220 + random.uniform(-30, 30)
        return f"Low: {low:.0f} PSI, High: {high:.0f} PSI"
    elif sensor_type == 'current':
        return f"{(18 + random.uniform(-8, 8)):.1f} Amps"
    elif sensor_type == 'voltage':
        return f"{(238 + random.uniform(-4, 4)):.1f}V"
    elif sensor_type == 'airflow':
        return f"{(450 + random.uniform(-100, 100)):.0f} CFM"
    elif sensor_type == 'resistance':
        return f"{(85 + random.uniform(-30, 30)):.0f} ohms"
    elif sensor_type == 'traffic':
        return random.choice(['1000 Mbps Full Duplex', '100 Mbps Full Duplex', 'Link Down'])
    elif sensor_type == 'errors':
        crc = random.uniform(0, 0.02)
        collisions = random.uniform(0, 10)
        return f"CRC: {crc:.3f}%, Collisions: {collisions:.1f}%"
    elif sensor_type == 'power':
        return f"{random.uniform(150, 300):.1f}W"
    else:
        return "N/A"

# --- Tool 1: Data Analysis Tool ---
def analyze_csv_data(query: str) -> str:
    try:
        ticket_file = next(f for f in glob.glob("*.csv") if "tickets" in f.lower())
    except StopIteration:
        return "No ticket CSV file was found."

    try:
        df = pd.read_csv(ticket_file)
        query_lower = query.lower()
        match = re.search(r"count of (.+)|how many (.+)", query_lower)
        if match:
            target_value = next(g for g in match.groups() if g is not None).strip().replace(" issues", "")
            target_column = None
            for col in df.columns:
                if "rca" in col.lower() or "root cause" in col.lower() or "summary" in col.lower():
                    target_column = col
                    break
            if not target_column:
                return "Could not determine a relevant column to search for in the CSV."

            count = df[df[target_column].str.contains(target_value, case=False, na=False)].shape[0]
            return f"Found **{count} tickets** related to **'{target_value}'** in `{os.path.basename(ticket_file)}`."
        return "I can count issues from ticket files. Try a query like: 'how many Device Faulty issues?'"
    except Exception as e:
        return f"An error occurred during CSV analysis: {e}"

# --- LLM Configuration ---
try:
    # IMPORTANT: Set your GEMINI_API_KEY in st.secrets
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    GEMINI_MODEL = None

def generate_response_stream(prompt_parts: List[Any]) -> Generator:
    if not GEMINI_MODEL:
        yield "The AI model is not configured. Please check your API key."
        return
    try:
        response_stream = GEMINI_MODEL.generate_content(prompt_parts, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"An error occurred while generating the response: {e}"

# --- Document Processing (Unchanged) ---
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

def smart_chunking(text: str, chunk_size: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk: chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 30]

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception:
        return ""

def extract_text_from_email(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                try: return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                except: continue
    elif msg.get_content_type() == "text/plain":
        try: return msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')
        except: return ""
    return ""

def format_email_for_rag(msg) -> str:
    body = extract_text_from_email(msg)
    if not body: return ""
    return f"--- Email ---\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\nDate: {msg['Date']}\n{body.strip()}\n--- End Email ---"

@st.cache_resource
def load_and_process_documents():
    # Assumes documents are in a 'documents' subfolder.
    # Create this folder and place your files inside.
    script_dir = os.path.dirname(__file__)
    docs_path = os.path.join(script_dir, 'documents')
    
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox"]
    all_files = [f for pattern in file_patterns for f in glob.glob(os.path.join(docs_path, pattern), recursive=True)]
    docs, file_paths = [], []
    
    progress_bar = st.progress(0, text="Loading knowledge base...")
    
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        progress_bar.progress((i + 1) / len(all_files), text=f"Processing: {file_name}")
        content = ""
        try:
            if file_path.endswith('.pdf'):
                content = extract_text_from_pdf(file_path)
            elif file_path.endswith('.eml'):
                with open(file_path, 'rb') as f:
                    content = format_email_for_rag(BytesParser(policy=policy.default).parse(f))
            elif file_path.endswith('.mbox'):
                mbox_content = [format_email_for_rag(msg) for msg in mailbox.mbox(file_path)]
                content = "\n\n".join(filter(None, mbox_content))
            elif file_path.endswith(('.txt', '.md', '.csv')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
        except Exception:
            continue
        
        if content and len(content.strip()) > 50:
            docs.append(clean_text(content))
            file_paths.append(file_path)
    
    progress_bar.empty()
    if docs:
        st.success(f"Knowledge base loaded with {len(docs)} documents.")
    return docs, file_paths

@st.cache_resource
def create_search_index(_documents, _file_paths):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    all_chunks, chunk_metadata = [], []
    
    for i, doc in enumerate(_documents):
        chunks = smart_chunking(doc, config.chunk_size)
        all_chunks.extend(chunks)
        chunk_metadata.extend([{'path': _file_paths[i]}] * len(chunks))
    
    if not all_chunks: return None, None, [], []
    
    with st.spinner("Embedding documents..."):
        embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    st.success(f"✅ Search index created with {len(all_chunks)} text chunks.")
    return index, model, all_chunks, chunk_metadata

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    if not query or index is None: return []
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and scores[0][i] > config.similarity_threshold:
             results.append({'content': chunks[idx], 'source': metadata[idx]['path'], 'similarity': scores[0][i]})
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# --- Advanced Diagnostic Functions ---
def detect_equipment_from_schema(filename: str, image_data=None) -> str:
    name = filename.lower()
    if 'hvac' in name or 'cooling' in name or 'air' in name:
        return 'hvac'
    if 'electrical' in name or 'power' in name or 'panel' in name:
        return 'electrical'
    if 'network' in name or 'switch' in name or 'router' in name:
        return 'network'
    
    # Fallback for demo purposes
    return random.choice(['hvac', 'electrical', 'network'])

def start_detailed_diagnostic(issue_description: str, equipment_type: str) -> Dict:
    if equipment_type not in EQUIPMENT_DATABASE:
        return {'error': f"Equipment type '{equipment_type}' not found in database"}
    
    equipment = EQUIPMENT_DATABASE[equipment_type]
    issue_lower = issue_description.lower()
    
    matched_issue = None
    matched_key = None
    
    for key, issue in equipment['detailed_issues'].items():
        if key in issue_lower or any(symptom in issue_lower for symptom in issue['symptoms']):
            matched_issue = issue
            matched_key = key
            break
            
    if not matched_issue:
        available_issues = list(equipment['detailed_issues'].keys())
        return {
            'error': f"Issue '{issue_description}' not found in diagnostic database",
            'available_issues': available_issues
        }
    
    return {
        'equipment_type': equipment_type,
        'issue_key': matched_key,
        'issue_data': matched_issue,
        'current_step': 0,
        'total_steps': len(matched_issue['diagnostic_flow'])
    }

def get_diagnostic_step_info(diagnostic_session: Dict, step_number: int) -> Dict:
    if step_number >= len(diagnostic_session['issue_data']['diagnostic_flow']):
        return {'error': 'Step number out of range'}
    
    step = diagnostic_session['issue_data']['diagnostic_flow'][step_number]
    
    sensor_data = {}
    if 'sensors' in step:
        for sensor in step['sensors']:
            sensor_data[sensor] = generate_sensor_data(sensor, diagnostic_session['equipment_type'])
    
    return {
        'step': step,
        'sensor_data': sensor_data,
        'step_number': step_number + 1,
        'total_steps': diagnostic_session['total_steps']
    }

# --- Initialize Session State ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "🔧 **Advanced Equipment Diagnostic AI Ready!**\n\nI provide real-time, step-by-step diagnostics with live sensor monitoring. Upload a schema diagram and describe your equipment issue to begin detailed analysis.", "timestamp": datetime.now()}
        ]
    
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    
    if "diagnostic_session" not in st.session_state:
        st.session_state.diagnostic_session = None
    
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    
    if "sensor_data" not in st.session_state:
        st.session_state.sensor_data = {}
    
    if "is_diagnostic_active" not in st.session_state:
        st.session_state.is_diagnostic_active = False

# --- Main App Interface ---
def main():
    st.title("🔧 Advanced Equipment Diagnostic AI")
    st.markdown("*Real-time troubleshooting with live sensor monitoring, powered by Gemini & RAG*")
    
    initialize_session_state()
    
    # Initialize systems
    if "init_done" not in st.session_state:
        st.session_state.init_done = False
        try:
            st.session_state.maintenance_pipeline = MaintenancePipeline()
            docs, paths = load_and_process_documents()
            if docs:
                search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths)
                st.session_state.search_args = {
                    'index': search_index,
                    'model': model,
                    'chunks': all_chunks,
                    'metadata': chunk_meta
                }
            else:
                st.session_state.search_args = None
            st.session_state.init_done = True
        except Exception as e:
            st.error(f"Initialization error: {e}")
            st.session_state.search_args = None
    
    # Sidebar Controls
    with st.sidebar:
        st.header("🛠️ Diagnostic Controls")
        
        st.subheader("📋 Schema Analysis")
        uploaded_file = st.file_uploader("Upload Schema Diagram", type=['png', 'jpg', 'jpeg', 'pdf'])
        
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Schema Diagram", use_column_width=True)
                
                if st.button("🔍 Analyze Schema", use_container_width=True):
                    with st.spinner("Analyzing schema diagram..."):
                        equipment_type = detect_equipment_from_schema(uploaded_file.name, image)
                        equipment = EQUIPMENT_DATABASE[equipment_type]
                        
                        st.session_state.current_analysis = {
                            'equipment_type': equipment_type,
                            'equipment_name': equipment['name'],
                            'components': equipment['components'],
                            'sensors': equipment['sensors'],
                            'confidence': random.randint(85, 95)
                        }
                        
                        analysis_msg = f"""🔍 **Schema Analysis Complete!**
                        
**Equipment Detected**: {equipment['name']}
**Confidence**: {st.session_state.current_analysis['confidence']}%

**Ready for diagnostics!** Describe your specific issue in the chat below."""
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": analysis_msg,
                            "timestamp": datetime.now(),
                            "type": "analysis"
                        })
                        st.rerun()
        
        st.subheader("🚨 Alert Simulation")
        if st.button("Generate Real-Time Alert", use_container_width=True):
            if hasattr(st.session_state, 'maintenance_pipeline'):
                alert = st.session_state.maintenance_pipeline.simulate_real_time_alert()
                
                alert_msg = f"""🚨 **REAL-TIME ALERT DETECTED**

**Device**: `{alert['device_id']}` ({alert['device_type']})
**Issue**: {alert['alert_type']}
**Location**: {alert['location']}

Initiating automated diagnostic protocol..."""
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": alert_msg,
                    "timestamp": datetime.now(),
                    "type": "alert"
                })
                st.rerun()
        
        if st.session_state.is_diagnostic_active:
            st.subheader("🔧 Active Diagnostic")
            if st.session_state.diagnostic_session:
                progress = (st.session_state.current_step + 1) / st.session_state.diagnostic_session['total_steps']
                st.progress(progress, f"Step {st.session_state.current_step + 1} of {st.session_state.diagnostic_session['total_steps']}")
            
            if st.button("⏭️ Next Step", use_container_width=True):
                if st.session_state.diagnostic_session and st.session_state.current_step < st.session_state.diagnostic_session['total_steps'] - 1:
                    st.session_state.current_step += 1
                    # Display the next step immediately
                    step_info = get_diagnostic_step_info(st.session_state.diagnostic_session, st.session_state.current_step)
                    st.session_state.sensor_data = step_info.get('sensor_data', {})
                    next_step_msg = f"""**Step {step_info['step_number']} of {step_info['total_steps']}**: {step_info['step']['action']}

**Instruction**: {step_info['step']['instruction']}
**Expected Result**: {step_info['step']['expected_result']}
"""
                    st.session_state.messages.append({"role": "assistant", "content": next_step_msg, "timestamp": datetime.now()})
                    st.rerun()
                else: # Last step was completed
                    st.session_state.is_diagnostic_active = False
                    st.session_state.diagnostic_session = None
                    st.session_state.sensor_data = {}
                    st.session_state.messages.append({"role": "assistant", "content": "🎉 **Diagnostic Complete!** All steps finished.", "timestamp": datetime.now()})
                    st.rerun()

            if st.button("🔄 Restart Diagnostic", use_container_width=True):
                st.session_state.diagnostic_session = None
                st.session_state.current_step = 0
                st.session_state.is_diagnostic_active = False
                st.session_state.sensor_data = {}
                st.session_state.messages.append({"role": "assistant", "content": "Diagnostic session has been reset.", "timestamp": datetime.now()})
                st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Diagnostic Chat")
        chat_container = st.container(height=500)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "timestamp" in msg:
                        st.caption(f"_{msg['timestamp'].strftime('%H:%M:%S')}_")

    with col2:
        if st.session_state.current_analysis:
            st.subheader("📋 Equipment Analysis")
            st.metric("Equipment", st.session_state.current_analysis['equipment_name'])
            st.metric("Confidence", f"{st.session_state.current_analysis['confidence']}%")
        
        if st.session_state.sensor_data:
            st.subheader("📊 Live Sensor Data")
            for sensor, value in st.session_state.sensor_data.items():
                st.metric(sensor.title(), value)
            
            # Auto-refresh sensor data
            if st.session_state.is_diagnostic_active:
                time.sleep(3)
                step_info = get_diagnostic_step_info(st.session_state.diagnostic_session, st.session_state.current_step)
                st.session_state.sensor_data = step_info.get('sensor_data', {})
                st.rerun()


    # Chat input is always at the bottom
    if prompt := st.chat_input("Describe the issue or ask a question..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })

        response = ""
        # If a schema has been analyzed and no diagnostic is active, this prompt starts the diagnostic
        if st.session_state.current_analysis and not st.session_state.is_diagnostic_active:
            diagnostic_result = start_detailed_diagnostic(
                prompt,
                st.session_state.current_analysis['equipment_type']
            )

            if 'error' not in diagnostic_result:
                st.session_state.diagnostic_session = diagnostic_result
                st.session_state.is_diagnostic_active = True
                st.session_state.current_step = 0

                step_info = get_diagnostic_step_info(diagnostic_result, 0)
                st.session_state.sensor_data = step_info.get('sensor_data', {})

                response = f"""🔧 **Diagnostic Initiated: {diagnostic_result['issue_key'].title()}**

**Step 1 of {diagnostic_result['total_steps']}**: {step_info['step']['action']}
**Instruction**: {step_info['step']['instruction']}
**Expected Result**: {step_info['step']['expected_result']}
"""
            else:
                response = f"❌ {diagnostic_result['error']}.\nAvailable issues: " + ", ".join(diagnostic_result.get('available_issues', []))
        
        # If no diagnostic context, fall back to RAG/AI Search
        else:
            if st.session_state.search_args:
                search_results = search_documents(prompt, **st.session_state.search_args)
                if search_results:
                    relevant_info = "\n\n".join([f"From {os.path.basename(r['source'])}:\n{r['content']}" for r in search_results[:2]])
                    prompt_parts = ["Based on this info:\n", relevant_info, "\n\nRespond to this query:\n", prompt]
                    # Since write_stream can't be used directly here, we aggregate the response
                    response = "".join(list(generate_response_stream(prompt_parts)))
                else:
                    response = "I couldn't find any relevant information in the knowledge base. Please try rephrasing or upload a schema to start a specific diagnostic."
            else:
                response = "The knowledge base is not loaded. Please upload a schema to begin."

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        st.rerun()

if __name__ == "__main__":
    main()
