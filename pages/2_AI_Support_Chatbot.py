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
    # This class can be expanded later if needed, but for now, the main logic is in the handler
    
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
            "alert_type": alert_type, "device_id": device, "device_type": device_info['type'],
            "location": device_info['location'], "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": f"{alert_type} detected on {device_info['type']} at {device_info['location']}"
        }

# --- LLM Configuration ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"❌ Error configuring Gemini API: {e}. Please set your API key in Streamlit secrets.")
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

# --- Document Processing & RAG Functions ---
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

@st.cache_resource
def load_and_process_documents():
    script_dir = os.path.dirname(__file__)
    docs_path = os.path.join(script_dir, 'documents')
    if not os.path.exists(docs_path):
        st.warning(f"Knowledge Base folder not found at {docs_path}. Create a 'documents' folder next to app.py to enable RAG.")
        return [], []
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox"]
    all_files = [f for p in file_patterns for f in glob.glob(os.path.join(docs_path, p), recursive=True)]
    docs, file_paths = [], []
    if all_files:
        progress_bar = st.progress(0, text="Loading knowledge base...")
        for i, file_path in enumerate(all_files):
            file_name = os.path.basename(file_path)
            progress_bar.progress((i + 1) / len(all_files), text=f"Processing: {file_name}")
            content = ""
            try:
                if file_path.endswith('.pdf'):
                    with fitz.open(file_path) as doc: content = "".join(page.get_text() for page in doc)
                elif file_path.endswith('.eml'):
                    with open(file_path, 'rb') as f:
                        msg = BytesParser(policy=policy.default).parse(f)
                        body = msg.get_body(preferencelist=('plain')).get_content() if msg else ''
                        content = f"From: {msg['From']}\nSubject: {msg['Subject']}\n{body}"
                elif file_path.endswith(('.txt', '.md', '.csv')):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            except Exception as e: st.warning(f"Could not process {file_name}: {e}")
            if content and len(content.strip()) > 50: docs.append(clean_text(content)); file_paths.append(file_path)
        progress_bar.empty()
    if docs: st.success(f"📚 Knowledge base loaded: {len(docs)} documents")
    else: st.info("📝 No documents found in 'documents' folder.")
    return docs, file_paths

@st.cache_resource
def create_search_index(_documents, _file_paths):
    if not _documents: return None, None, [], []
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        all_chunks = [chunk for doc in _documents for chunk in smart_chunking(doc, config.chunk_size)]
        if not all_chunks: return None, None, [], []
        with st.spinner("🔍 Creating search embeddings..."):
            embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        st.success(f"✅ Search index ready: {len(all_chunks)} chunks")
        # Store file paths corresponding to chunks for citation (simplified mapping)
        chunk_metadata = [{'source': path} for path, doc in zip(_file_paths, _documents) for _ in smart_chunking(doc, config.chunk_size)]
        return index, model, all_chunks, chunk_metadata
    except Exception as e:
        st.error(f"Error creating search index: {e}")
        return None, None, [], []

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    if not query or index is None: return []
    try:
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
        results = [{'content': chunks[idx], 'source': metadata[idx]['source'], 'similarity': scores[0][i]}
                   for i, idx in enumerate(indices[0]) if idx != -1 and scores[0][i] > config.similarity_threshold]
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def detect_equipment_from_schema(filename: str, image_data=None) -> Dict:
    # This function remains the same, with vision capabilities
    filename_lower = filename.lower()
    equipment_mapping = {
        'hvac': ['hvac', 'cooling', 'heating'], 'electrical': ['electrical', 'power', 'panel'],
        'network': ['network', 'switch', 'router'], 'server': ['server', 'cpu', 'rack'],
        'industrial': ['motor', 'pump', 'valve']
    }
    detected_type = 'general'
    for eq_type, keywords in equipment_mapping.items():
        if any(keyword in filename_lower for keyword in keywords):
            detected_type = eq_type; break
    if image_data and GEMINI_MODEL:
        try:
            prompt = """Analyze the provided schema image. Identify the primary equipment type and list key components/instruments that are common points of failure. Respond ONLY with a valid JSON object: {"equipment_type": "...", "identified_components": ["..."]}"""
            response = GEMINI_MODEL.generate_content([prompt, image_data])
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                vision_analysis = json.loads(json_match.group())
                final_type = vision_analysis.get('equipment_type', detected_type).lower()
                equipment_info = EQUIPMENT_KNOWLEDGE.get(final_type, {})
                return {'equipment_type': final_type, 'equipment_name': equipment_info.get('name', 'Unknown'),
                        'components': vision_analysis.get('identified_components', []),
                        'common_issues': equipment_info.get('common_issues', []), 'confidence': 95}
        except Exception as e: st.warning(f"Vision analysis failed: {e}. Falling back to filename analysis.")
    equipment_info = EQUIPMENT_KNOWLEDGE.get(detected_type, {})
    return {'equipment_type': detected_type, 'equipment_name': equipment_info.get('name', 'General'),
            'components': equipment_info.get('components', []), 'common_issues': equipment_info.get('common_issues', []),
            'confidence': 60}

# --- Session State Initialization ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "🔧 **Smart Equipment Diagnostic AI Ready!**\nHow can I help you troubleshoot today?", "timestamp": datetime.now()}]
    defaults = {"current_analysis": None, "diagnostic_engine": None, "maintenance_pipeline": None}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

# --- NEW: Enhanced Chat Response Logic ---
def get_kb_context(query: str, search_args) -> List[Dict]:
    """Search the knowledge base and return top relevant chunks."""
    if not search_args: return []
    return search_documents(query, **search_args)

def format_kb_citations(results: List[Dict]) -> str:
    """Format KB results for display as expandable citations."""
    if not results: return ""
    citations = [f"<details><summary><b>📄 Source {i+1}: {os.path.basename(res['source'])} (score: {res['similarity']:.2f})</b></summary>{res['content']}</details>"
                 for i, res in enumerate(results)]
    return "\n".join(citations)

def enhanced_chat_response(user_query: str):
    """Handles the entire chat response pipeline: RAG search, LLM call, and streaming."""
    # 1. Search Knowledge Base
    kb_results = get_kb_context(user_query, st.session_state.get("search_args"))
    kb_context = "\n\n".join([f"[Source {i+1}] {r['content']}" for i, r in enumerate(kb_results)])
    citations_md = format_kb_citations(kb_results)

    # 2. Compose LLM prompt with diagnostic instructions and context
    prompt = f"""You are a world-class equipment diagnostic assistant.
Your goal is to help users solve technical problems step-by-step.
If the answer is available in the knowledge base, use it and cite the source number (e.g., [1]) at the end of the relevant sentence.

Knowledge Base Context:
{kb_context if kb_context else '[No relevant context found]'}

User Query:
{user_query}

Provide a clear, actionable, and safe diagnostic response.
"""

    # 3. Stream LLM response
    if 'live_response' not in st.session_state:
        st.session_state.live_response = ""
    
    response_placeholder = st.empty()
    full_response = ""
    for chunk in generate_response_stream(prompt):
        full_response += chunk
        with response_placeholder.container():
             st.chat_message("assistant").markdown(full_response + " ▌", unsafe_allow_html=True)
    
    # 4. Finalize response and add to history
    final_content = full_response + (f"\n\n---\n**Sources:**\n{citations_md}" if citations_md else "")
    st.session_state.messages.append({"role": "assistant", "content": final_content, "timestamp": datetime.now()})
    
    # Clean up and rerun
    if 'live_response' in st.session_state:
        del st.session_state['live_response']
    response_placeholder.empty()
    st.rerun()

# --- Main Application ---
def main():
    st.markdown("""<style> .stTabs [data-baseweb="tab-list"] {margin-bottom: 0;} </style>""", unsafe_allow_html=True)
    initialize_session_state()

    if "welcome_shown" not in st.session_state:
        st.balloons(); st.session_state.welcome_shown = True

    if GEMINI_MODEL and not st.session_state.diagnostic_engine:
        st.session_state.diagnostic_engine = SmartDiagnosticEngine(GEMINI_MODEL)

    if "init_done" not in st.session_state:
        with st.spinner("🚀 Initializing Smart Diagnostic AI..."):
            st.session_state.maintenance_pipeline = MaintenancePipeline()
            docs, paths = load_and_process_documents()
            if docs:
                search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths)
                st.session_state.search_args = {'index': search_index, 'model': model, 'chunks': all_chunks, 'metadata': chunk_meta}
            else: st.session_state.search_args = None
            st.session_state.init_done = True

    st.markdown(f"""<div style="background:#e3f2fd;padding:0.7em 1em;border-radius:10px;margin-bottom:1em;">
        <b>Status:</b> <span style="color:green">Online</span> &nbsp;|&nbsp;
        <b>Knowledge Base:</b> {"Loaded" if st.session_state.get("search_args") else "Not Loaded"} &nbsp;|&nbsp;
        <b>Gemini AI:</b> {"Connected" if GEMINI_MODEL else "Not Connected"}
        </div>""", unsafe_allow_html=True)

    tabs = st.tabs(["💬 Chat", "🛠️ Maintenance Overview"])

    # --- Chat Tab ---
    with tabs[0]:
        st.subheader("Smart Diagnostic Chat")
        if st.button("🧹 Clear Chat", help="Clear all chat history"):
            st.session_state.messages = st.session_state.messages[:1]
            st.rerun()

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                st.caption(f"_{msg['timestamp'].strftime('%H:%M:%S')}_")

        if prompt := st.chat_input("Describe your equipment issue or ask any technical question..."):
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now()})
            st.rerun() # Show user message immediately

    # Handle response generation after user message is displayed
    if st.session_state.messages[-1]["role"] == "user":
        enhanced_chat_response(st.session_state.messages[-1]["content"])

    # --- Maintenance Overview Tab ---
    with tabs[1]:
        st.subheader("Maintenance Overview")
        if st.session_state.maintenance_pipeline:
            df = pd.DataFrame(st.session_state.maintenance_pipeline.maintenance_data).T
            st.dataframe(df, use_container_width=True, height=400)
            with st.expander("Show High Risk Equipment"):
                st.write(df[df['risk_level'] == 'HIGH'])
        else: st.info("Maintenance data not loaded.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("🛠️ Diagnostic Tools")
        st.markdown("Upload a schema or generate a real-time alert to start.")
        st.divider()
        st.subheader("📋 Schema Analysis")
        uploaded_file = st.file_uploader("Upload Equipment Schema/Diagram", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Schema", use_column_width=True)
            if st.button("🔍 Analyze Schema", use_container_width=True):
                with st.spinner("Analyzing equipment schema with vision..."):
                    analysis_result = detect_equipment_from_schema(uploaded_file.name, image)
                    st.session_state.current_analysis = analysis_result
                    components_list = analysis_result.get('components', [])
                    if components_list:
                        components_str = '\n'.join([f"- {comp}" for comp in components_list[:7]])
                        analysis_msg = f"🔍 **Vision Analysis Complete!**\n**Equipment Detected**: {analysis_result['equipment_name']}\n**Key Components Identified:**\n{components_str}\n\nDescribe any issues with these components."
                    else: analysis_msg = "Could not identify specific components from the schema. Please describe the issue."
                    st.session_state.messages.append({"role": "assistant", "content": analysis_msg, "timestamp": datetime.now()})
                    st.rerun()
        st.divider()
        st.subheader("🚨 Equipment Monitoring")
        if st.button("Generate Alert", use_container_width=True):
            if st.session_state.maintenance_pipeline:
                alert = st.session_state.maintenance_pipeline.simulate_real_time_alert()
                alert_msg = f"🚨 **EQUIPMENT ALERT**\n**Device**: {alert['device_id']}\n**Issue**: {alert['alert_type']}\n**Severity**: {alert['severity']}\n\nWould you like diagnostic assistance for this issue?"
                st.session_state.messages.append({"role": "assistant", "content": alert_msg, "timestamp": datetime.now()})
                st.rerun()
        st.divider()
        if st.session_state.current_analysis:
            st.subheader("📊 Current Analysis")
            with st.container(border=True):
                st.metric("Equipment", st.session_state.current_analysis.get('equipment_name', 'N/A'))
                st.metric("Confidence", f"{st.session_state.current_analysis.get('confidence', 0)}%")
                with st.expander("Identified Components"):
                    for comp in st.session_state.current_analysis.get('components', []): st.write(f"• {comp}")

if __name__ == "__main__":
    main()
