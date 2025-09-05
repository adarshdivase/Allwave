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
from typing import List, Dict, Any, Generator, Optional
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
    page_title="Smart Equipment Diagnostic AI",
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

# --- Smart Equipment Knowledge Base ---
EQUIPMENT_KNOWLEDGE = {
    'hvac': {
        'name': 'HVAC System',
        'components': ['compressor', 'condenser', 'evaporator', 'expansion valve', 'refrigerant lines', 'thermostat', 'blower motor', 'air filter', 'ductwork'],
        'common_issues': ['no cooling', 'poor airflow', 'strange noises', 'high energy bills', 'water leaks', 'frozen coils', 'thermostat issues']
    },
    'electrical': {
        'name': 'Electrical System',
        'components': ['circuit breakers', 'outlets', 'switches', 'wiring', 'panels', 'meters', 'transformers'],
        'common_issues': ['power outages', 'flickering lights', 'tripped breakers', 'burning smell', 'shock hazards', 'overheating']
    },
    'network': {
        'name': 'Network Equipment',
        'components': ['switches', 'routers', 'cables', 'ports', 'access points', 'firewalls'],
        'common_issues': ['connection timeout', 'slow speeds', 'intermittent connectivity', 'device offline', 'high latency']
    },
    'server': {
        'name': 'Server Hardware',
        'components': ['cpu', 'memory', 'storage', 'power supply', 'cooling fans', 'motherboard'],
        'common_issues': ['high cpu usage', 'memory errors', 'disk failures', 'overheating', 'boot failures', 'network issues']
    },
    'industrial': {
        'name': 'Industrial Equipment',
        'components': ['motors', 'pumps', 'valves', 'sensors', 'controllers', 'drives'],
        'common_issues': ['motor failures', 'pump problems', 'sensor malfunctions', 'control system errors', 'mechanical wear']
    }
}

# --- Smart Diagnostic Engine ---
class SmartDiagnosticEngine:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        self.diagnostic_history = []
    
    def analyze_user_query(self, query: str, equipment_context: Optional[Dict] = None) -> Dict:
        """Intelligently analyze user query to understand the problem and equipment type"""
        system_prompt = """You are an expert equipment diagnostic AI. Analyze the user's query and extract:

1. Equipment type (hvac, electrical, network, server, industrial, etc.)
2. Specific component mentioned (if any)
3. Problem description
4. Urgency level (low, medium, high, critical)
5. Keywords that indicate the issue type

Respond in JSON format:
{
    "equipment_type": "detected equipment type",
    "component": "specific component or null",
    "problem_summary": "brief problem description",
    "urgency": "urgency level",
    "keywords": ["keyword1", "keyword2"],
    "confidence": "confidence percentage as integer"
}

Examples:
- "HVAC not cooling" -> equipment_type: "hvac", problem_summary: "cooling failure"
- "Server won't boot" -> equipment_type: "server", problem_summary: "boot failure"
- "Network switch port dead" -> equipment_type: "network", component: "port", problem_summary: "port failure"
"""
        try:
            response = self.gemini_model.generate_content(
                f"{system_prompt}\n\nUser Query: {query}"
            )
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                return self._fallback_analysis(query)
        except Exception as e:
            st.error(f"Error in query analysis: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict:
        query_lower = query.lower()
        equipment_type = "general"
        for eq_type, eq_data in EQUIPMENT_KNOWLEDGE.items():
            if any(keyword in query_lower for keyword in [eq_type, eq_data['name'].lower()]):
                equipment_type = eq_type
                break
        urgency = "medium"
        if any(word in query_lower for word in ['urgent', 'critical', 'emergency', 'fire', 'smoke']):
            urgency = "critical"
        elif any(word in query_lower for word in ['broken', 'failed', 'dead', 'not working']):
            urgency = "high"
        return {
            "equipment_type": equipment_type,
            "component": None,
            "problem_summary": query[:100],
            "urgency": urgency,
            "keywords": query_lower.split()[:5],
            "confidence": 70
        }
    
    def generate_diagnostic_solution(self, query: str, analysis: Dict, rag_context: str = "") -> str:
        equipment_info = EQUIPMENT_KNOWLEDGE.get(analysis.get('equipment_type', 'general'), {})
        diagnostic_prompt = f"""You are an expert equipment diagnostic technician. Provide a comprehensive diagnostic solution for the following issue:

**Equipment Type**: {analysis.get('equipment_type', 'Unknown')}
**Problem**: {analysis.get('problem_summary', query)}
**Urgency**: {analysis.get('urgency', 'medium')}
**Component**: {analysis.get('component', 'Not specified')}

**Equipment Knowledge**:
- Components: {equipment_info.get('components', [])}
- Common Issues: {equipment_info.get('common_issues', [])}

**Additional Context from Knowledge Base**:
{rag_context}

**User's Original Query**: {query}

Please provide:

1. **Immediate Safety Checks** (if applicable)
2. **Diagnostic Steps** (step-by-step troubleshooting)
3. **Possible Causes** (ranked by likelihood)
4. **Solutions** (detailed repair/fix instructions)
5. **Tools/Parts Needed**
6. **Prevention Tips**
7. **When to Call Professional** (if applicable)

Format your response in clear sections with actionable steps. Be specific and practical."""
        try:
            response = self.gemini_model.generate_content(diagnostic_prompt)
            return response.text
        except Exception as e:
            return f"Error generating solution: {e}\n\nPlease try rephrasing your question or contact technical support."
    
    def generate_followup_questions(self, query: str, analysis: Dict) -> List[str]:
        equipment_type = analysis.get('equipment_type', 'general')
        try:
            followup_prompt = f"""Based on this equipment issue, generate 3-4 specific diagnostic questions to better understand the problem:

Equipment Type: {equipment_type}
Issue: {analysis.get('problem_summary', query)}

Generate practical questions that a technician would ask to diagnose the issue. Return only the questions, one per line, starting with '- '."""
            response = self.gemini_model.generate_content(followup_prompt)
            questions = [line.strip()[2:] for line in response.text.split('\n') if line.strip().startswith('- ')]
            return questions[:4]
        except:
            return self._get_fallback_questions(equipment_type)
    
    def _get_fallback_questions(self, equipment_type: str) -> List[str]:
        common_questions = [
            "When did the problem first occur?",
            "Are there any error messages or warning lights?",
            "Has anything changed recently (maintenance, updates, etc.)?",
            "Is the problem constant or intermittent?"
        ]
        equipment_specific = {
            'hvac': ["What is the current temperature reading?", "Are all vents blowing air?"],
            'electrical': ["Are any circuit breakers tripped?", "Do you smell anything burning?"],
            'network': ["Are the status LEDs on the device lit?", "Can you ping the device?"],
            'server': ["What error messages appear during boot?", "Are the fans running?"]
        }
        return common_questions + equipment_specific.get(equipment_type, [])

# --- Enhanced Maintenance Pipeline ---
class MaintenancePipeline:
    def __init__(self):
        self.maintenance_data = self._load_maintenance_data()
        self.equipment_list = list(self.maintenance_data.keys())

    def _load_maintenance_data(self) -> Dict:
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'Network Switch', 'Server', 'Industrial Motor']
        for i in range(30):
            eq_type = random.choice(equipment_types)
            fail_prob = random.uniform(0.1, 0.9)
            equipment_data[f"{eq_type.replace(' ','_')}_{i+1}"] = {
                'type': eq_type,
                'location': f"Building A - Floor {random.randint(1,4)} - Rack {chr(65+i%4)}",
                'failure_probability': fail_prob,
                'risk_level': 'HIGH' if fail_prob > 0.7 else 'MEDIUM' if fail_prob > 0.4 else 'LOW',
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 5000),
                'last_issue': random.choice(['Overheating', 'Power failure', 'Network timeout', 'Mechanical wear', 'Software error'])
            }
        return equipment_data

    def simulate_real_time_alert(self) -> Dict:
        alert_types = [
            "Device Offline", "High Temperature", "High CPU Usage", 
            "Memory Error", "Disk Failure", "Network Timeout",
            "Power Supply Failure", "Cooling Fan Error", "Configuration Error"
        ]
        alert_type = random.choice(alert_types)
        device = random.choice(self.equipment_list)
        device_info = self.maintenance_data[device]
        return {
            "alert_type": alert_type,
            "device_id": device,
            "device_type": device_info['type'],
            "location": device_info['location'],
            "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"{alert_type} detected on {device_info['type']} at {device_info['location']}"
        }

# --- Tool: Data Analysis Tool ---
def analyze_csv_data(query: str) -> str:
    try:
        csv_files = glob.glob("*.csv")
        if not csv_files:
            return "No CSV files found in the current directory."
        
        ticket_file = None
        for file in csv_files:
            if any(keyword in file.lower() for keyword in ['ticket', 'maintenance', 'issue', 'problem']):
                ticket_file = file
                break
        
        if not ticket_file:
            ticket_file = csv_files[0]

        df = pd.read_csv(ticket_file)
        if 'GEMINI_MODEL' in globals() and GEMINI_MODEL:
            analysis_prompt = f"""Analyze this CSV data query: "{query}"

CSV columns: {list(df.columns)}
CSV shape: {df.shape}
Sample data: {df.head().to_string()}

Provide analysis based on the query. If it's asking for counts, provide the numbers. If asking for trends, describe them."""
            try:
                response = GEMINI_MODEL.generate_content(analysis_prompt)
                return f"**CSV Analysis Results:**\n\n{response.text}"
            except Exception as e:
                return f"Error in CSV analysis: {e}"
        return f"CSV file '{ticket_file}' loaded with {len(df)} records. Please specify what analysis you need."
    except Exception as e:
        return f"Error analyzing CSV data: {e}"

# --- LLM Configuration ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
    st.success("✅ Gemini AI connected successfully!")
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {e}")
    GEMINI_MODEL = None

def generate_response_stream(prompt: str) -> Generator[str, None, None]:
    if not GEMINI_MODEL:
        yield "❌ Gemini AI is not configured. Please check your API key in Streamlit secrets."
        return
    try:
        response = GEMINI_MODEL.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"❌ Error generating response: {e}"

# --- Document Processing Functions ---
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

def smart_chunking(text: str, chunk_size: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk: 
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk: 
        chunks.append(current_chunk.strip())
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
                try: 
                    return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                except: 
                    continue
    elif msg.get_content_type() == "text/plain":
        try: 
            return msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')
        except: 
            return ""
    return ""

def format_email_for_rag(msg) -> str:
    body = extract_text_from_email(msg)
    if not body: 
        return ""
    return f"--- Email ---\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\nDate: {msg['Date']}\n{body.strip()}\n--- End Email ---"

@st.cache_resource
def load_and_process_documents():
    script_dir = os.path.dirname(__file__)
    docs_path = os.path.join(script_dir, 'documents')
    if not os.path.exists(docs_path):
        st.warning(f"Documents folder not found at {docs_path}")
        return [], []
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox"]
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(os.path.join(docs_path, pattern), recursive=True))
    
    docs, file_paths = [], []
    if all_files:
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
            except Exception as e:
                st.warning(f"Could not process {file_name}: {e}")
                continue
            
            if content and len(content.strip()) > 50:
                docs.append(clean_text(content))
                file_paths.append(file_path)
        progress_bar.empty()

    if docs:
        st.success(f"📚 Knowledge base loaded: {len(docs)} documents")
    else:
        st.info("📝 No documents found. You can still use the AI for general equipment diagnostics.")
    return docs, file_paths

@st.cache_resource
def create_search_index(_documents, _file_paths):
    if not _documents:
        return None, None, [], []
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        all_chunks, chunk_metadata = [], []
        for i, doc in enumerate(_documents):
            chunks = smart_chunking(doc, config.chunk_size)
            all_chunks.extend(chunks)
            chunk_metadata.extend([{'path': _file_paths[i]}] * len(chunks))
        
        if not all_chunks: 
            return None, None, [], []
        
        with st.spinner("🔍 Creating search embeddings..."):
            embeddings = model.encode(all_chunks, show_progress_bar=False, normalize_embeddings=True)
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        st.success(f"✅ Search index ready: {len(all_chunks)} chunks")
        return index, model, all_chunks, chunk_metadata
    except Exception as e:
        st.error(f"Error creating search index: {e}")
        return None, None, [], []

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    if not query or index is None: 
        return []
    try:
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and scores[0][i] > config.similarity_threshold:
                results.append({
                    'content': chunks[idx], 
                    'source': metadata[idx]['path'], 
                    'similarity': scores[0][i]
                })
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# --- UPDATED: Equipment Detection from Schema (with Vision) ---
def detect_equipment_from_schema(filename: str, image_data=None) -> Dict:
    """
    Analyzes an uploaded schema using both filename and multimodal vision 
    to identify equipment type and key components.
    """
    # --- Step 1: Initial analysis based on filename (as a fallback) ---
    filename_lower = filename.lower()
    equipment_mapping = {
        'hvac': ['hvac', 'cooling', 'heating', 'air', 'ventilation', 'climate', 'chiller', 'boiler'],
        'electrical': ['electrical', 'power', 'panel', 'circuit', 'voltage', 'current', 'transformer'],
        'network': ['network', 'switch', 'router', 'ethernet', 'lan', 'wan', 'tcp', 'ip'],
        'server': ['server', 'cpu', 'memory', 'storage', 'rack', 'blade', 'datacenter'],
        'industrial': ['motor', 'pump', 'valve', 'conveyor', 'machine', 'industrial', 'manufacturing']
    }
    detected_type = 'general'
    for eq_type, keywords in equipment_mapping.items():
        if any(keyword in filename_lower for keyword in keywords):
            detected_type = eq_type
            break

    # --- Step 2: Multimodal analysis if an image is provided ---
    if image_data and GEMINI_MODEL:
        try:
            # Prepare the prompt for the vision model
            prompt = """
            You are an expert systems engineer specializing in equipment diagnostics. 
            Analyze the provided schema image. 
            
            1.  Identify the primary type of equipment shown (e.g., HVAC, Electrical Panel, Network Rack).
            2.  List the key components, instruments, or parts that are visible and are common points of failure.
            
            Respond ONLY with a valid JSON object in the following format:
            {
              "equipment_type": "The detected equipment type",
              "identified_components": ["Component 1", "Sensor X", "Valve Y", "Circuit Breaker Z"]
            }
            """
            
            # Send the prompt and image to the Gemini model
            response = GEMINI_MODEL.generate_content([prompt, image_data])
            
            # Extract and parse the JSON response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                vision_analysis = json.loads(json_match.group())
                
                # Combine vision analysis with filename analysis
                final_type = vision_analysis.get('equipment_type', detected_type).lower()
                equipment_info = EQUIPMENT_KNOWLEDGE.get(final_type, {})
                
                return {
                    'equipment_type': final_type,
                    'equipment_name': equipment_info.get('name', 'Unknown Equipment'),
                    'components': vision_analysis.get('identified_components', []),
                    'common_issues': equipment_info.get('common_issues', []),
                    'confidence': 95  # High confidence from vision analysis
                }
        except Exception as e:
            st.warning(f"Vision analysis failed: {e}. Falling back to filename analysis.")

    # --- Step 3: Fallback to filename-only analysis if vision fails or no image ---
    equipment_info = EQUIPMENT_KNOWLEDGE.get(detected_type, {})
    return {
        'equipment_type': detected_type,
        'equipment_name': equipment_info.get('name', 'General Equipment'),
        'components': equipment_info.get('components', ['various components']),
        'common_issues': equipment_info.get('common_issues', ['general malfunctions']),
        'confidence': 60
    }

# --- Session State Initialization ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """🔧 **Smart Equipment Diagnostic AI Ready!**

I'm your intelligent equipment diagnostic assistant. I can help you with:

• **Any equipment issue** - HVAC, electrical, network, servers, industrial equipment
• **Intelligent problem analysis** - I understand your problem description
• **Step-by-step solutions** - Detailed troubleshooting guides
• **Safety recommendations** - When to call professionals
• **Preventive maintenance** - Tips to avoid future issues

**Just describe your problem in plain English!** For example:
- "My air conditioner stopped working"
- "Server won't boot up"
- "Network switch port is dead"
- "Electrical panel keeps tripping"

How can I help you today? 🛠️""", 
                "timestamp": datetime.now()
            }
        ]
    defaults = {
        "current_analysis": None,
        "diagnostic_engine": None,
        "followup_questions": [],
        "maintenance_pipeline": None,
        "chat_history": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Refactored Response Handling Function ---
def handle_user_prompt(prompt: str):
    """Analyzes a prompt, gets a RAG context, and generates a diagnostic solution."""
    with st.spinner("🤖 Analyzing issue and generating solution..."):
        try:
            response_content = ""
            if st.session_state.diagnostic_engine:
                analysis = st.session_state.diagnostic_engine.analyze_user_query(
                    prompt,
                    st.session_state.current_analysis
                )
                
                # Update current analysis based on the new prompt
                if analysis.get('equipment_type') != 'general':
                     st.session_state.current_analysis = {
                        'equipment_name': EQUIPMENT_KNOWLEDGE.get(analysis['equipment_type'], {}).get('name', 'Unknown'),
                        'confidence': analysis.get('confidence', 75),
                        'components': EQUIPMENT_KNOWLEDGE.get(analysis['equipment_type'], {}).get('components', [])
                    }

                rag_context = ""
                if st.session_state.search_args:
                    search_results = search_documents(prompt, **st.session_state.search_args)
                    if search_results:
                        rag_context = "\n\n".join([
                            f"From {os.path.basename(r['source'])}:\n{r['content']}" 
                            for r in search_results[:2]
                        ])
                
                response_content = st.session_state.diagnostic_engine.generate_diagnostic_solution(
                    prompt, analysis, rag_context
                )
                
                st.session_state.followup_questions = st.session_state.diagnostic_engine.generate_followup_questions(
                    prompt, analysis
                )
            
            # This is a simplified fallback if the diagnostic engine isn't ready
            else:
                response_content = analyze_csv_data(prompt)
                if "No CSV files found" in response_content:
                    response_content = "The diagnostic engine is not available. Please check the Gemini API configuration."

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content,
                "timestamp": datetime.now()
            })
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ An error occurred while processing your request: {e}",
                "timestamp": datetime.now()
            })

# --- Main Application ---
def main():
    st.title("🔧 Smart Equipment Diagnostic AI")
    st.markdown("*Intelligent troubleshooting for any equipment - powered by Gemini AI & RAG*")
    
    initialize_session_state()

    # Initialize diagnostic engine
    if GEMINI_MODEL and not st.session_state.diagnostic_engine:
        st.session_state.diagnostic_engine = SmartDiagnosticEngine(GEMINI_MODEL)

    # Initialize systems
    if "init_done" not in st.session_state:
        with st.spinner("🚀 Initializing Smart Diagnostic AI..."):
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

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Diagnostic Tools")
        
        # Schema Upload
        st.subheader("📋 Schema Analysis")
        uploaded_file = st.file_uploader(
            "Upload Equipment Schema/Diagram", 
            type=['png', 'jpg', 'jpeg', 'pdf'],
            help="Upload a diagram to help identify your equipment"
        )
        if uploaded_file:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Schema", use_column_width=True)
                if st.button("🔍 Analyze Schema", use_container_width=True):
                    with st.spinner("Analyzing equipment schema with vision..."):
                        analysis_result = detect_equipment_from_schema(uploaded_file.name, image)
                        st.session_state.current_analysis = analysis_result
                        
                        # Create a more detailed message based on vision analysis
                        components_list = analysis_result.get('components', [])
                        if components_list:
                            components_str = '\n'.join([f"- {comp}" for comp in components_list[:7]])
                            analysis_msg = f"""🔍 **Vision Analysis Complete!**

**Equipment Detected**: {analysis_result['equipment_name']}
**Confidence**: {analysis_result['confidence']}%

**Key Components Identified from Schema:**
{components_str}

Describe any issues you're experiencing with these components."""
                        else:
                            analysis_msg = "Could not identify specific components from the schema. Please describe the issue."

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": analysis_msg,
                            "timestamp": datetime.now(),
                            "type": "analysis"
                        })
                        st.rerun()

        # Real-time Alerts
        st.subheader("🚨 Equipment Monitoring")
        if st.button("Generate Alert", use_container_width=True):
            if st.session_state.maintenance_pipeline:
                alert = st.session_state.maintenance_pipeline.simulate_real_time_alert()
                alert_msg = f"""🚨 **EQUIPMENT ALERT**

**Device**: {alert['device_id']}
**Issue**: {alert['alert_type']}
**Severity**: {alert['severity']}
**Location**: {alert['location']}

Would you like diagnostic assistance for this issue?"""
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": alert_msg,
                    "timestamp": datetime.now(),
                    "type": "alert"
                })
                st.rerun()

        # Current Analysis Display
        if st.session_state.current_analysis:
            st.subheader("📊 Current Analysis")
            with st.container(border=True):
                st.metric("Equipment", st.session_state.current_analysis.get('equipment_name', 'N/A'))
                st.metric("Confidence", f"{st.session_state.current_analysis.get('confidence', 0)}%")
                with st.expander("Identified Components"):
                    st.write("**Components:**")
                    for comp in st.session_state.current_analysis.get('components', [])[:10]:
                        st.write(f"• {comp}")
                        
        # Followup Questions Logic
        if st.session_state.followup_questions:
            st.subheader("🤔 Diagnostic Questions")
            st.write("*Click a question to get a more detailed diagnosis:*")
            for i, question in enumerate(st.session_state.followup_questions):
                if st.button(f"❓ {question}", key=f"q_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now()
                    })
                    handle_user_prompt(question)
                    st.session_state.followup_questions = []
                    st.rerun()

    # Main chat interface
    st.subheader("💬 Smart Diagnostic Chat")
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "timestamp" in msg:
                    st.caption(f"_{msg['timestamp'].strftime('%H:%M:%S')}_")
    
    # Chat Input Logic
    if prompt := st.chat_input("Describe your equipment issue or ask any technical question..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        handle_user_prompt(prompt)
        st.rerun()

if __name__ == "__main__":
    main()
