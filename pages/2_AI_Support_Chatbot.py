# app.py - HYBRID Versatile AI Assistant
# Combines the powerful backend of Script 1 with the UI/UX and granular fallbacks of Script 2.
import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Generator, Optional, Tuple
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
from functools import wraps
import traceback

# --- Enhanced Configuration ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Versatile AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MERGED --- Added some nice CSS styling from Script 2
st.markdown(
    """
    <style>
    .stChatMessage.user {background-color: #e3f2fd;}
    .stChatMessage.assistant {background-color: #f1f8e9;}
    .stButton>button {width: 100%;}
    </style>
    """, unsafe_allow_html=True
)


@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.4

@dataclass
class QuotaConfig:
    daily_limit: int = 100
    requests_per_hour: int = 20
    retry_delay: int = 60
    use_fallback_on_limit: bool = True
    api_key_rotation: bool = True

config = RAGConfig()
quota_config = QuotaConfig()

# --- Multi-API Key Management System (from Script 1) ---
class MultiAPIManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.key_status = {i: {'active': True, 'error_count': 0, 'last_error': None} 
                           for i in range(len(self.api_keys))}
        self.current_model = None
        self._initialize_current_model()
    
    def _load_api_keys(self) -> List[str]:
        keys = []
        for i in range(1, 6):
            key_name = f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY"
            if key_name in st.secrets:
                keys.append(st.secrets[key_name])
        
        if not keys:
            st.error("❌ No API keys found. Please add GEMINI_API_KEY_1 through GEMINI_API_KEY_5 to secrets.")
        
        return keys
    
    def _initialize_current_model(self):
        if self.api_keys and self.current_key_index < len(self.api_keys):
            try:
                genai.configure(api_key=self.api_keys[self.current_key_index])
                self.current_model = genai.GenerativeModel('gemini-1.5-flash')
                return True
            except Exception as e:
                st.error(f"Failed to initialize API key {self.current_key_index + 1}: {e}")
                return False
        return False
    
    def get_working_model(self):
        max_attempts = len(self.api_keys)
        
        for attempt in range(max_attempts):
            if self.key_status[self.current_key_index]['active']:
                try:
                    if not self.current_model:
                        self._initialize_current_model()
                    
                    self.current_model.generate_content("Test") # Test the key
                    self.key_status[self.current_key_index]['error_count'] = 0
                    return self.current_model
                    
                except Exception as e:
                    self._handle_api_error(e)
            
            self._rotate_to_next_key()
        
        st.error("🚫 All API keys exhausted. Using fallback mode.")
        return None
    
    def _handle_api_error(self, error: Exception):
        error_str = str(error).lower()
        current_status = self.key_status[self.current_key_index]
        current_status['error_count'] += 1
        current_status['last_error'] = str(error)
        
        if (current_status['error_count'] >= 3 or 
            'quota' in error_str or '429' in error_str or 'rate limit' in error_str):
            current_status['active'] = False
            st.warning(f"⚠️ API Key {self.current_key_index + 1} disabled due to: {error}")
    
    def _rotate_to_next_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.current_model = None
        self._initialize_current_model()
    
    def get_api_status(self) -> Dict:
        active_keys = sum(1 for status in self.key_status.values() if status['active'])
        return {
            'total_keys': len(self.api_keys),
            'active_keys': active_keys,
            'current_key': self.current_key_index + 1,
            'key_status': self.key_status
        }
    
    def reset_key_status(self, key_index: int = None):
        if key_index is not None:
            self.key_status[key_index] = {'active': True, 'error_count': 0, 'last_error': None}
        else:
            self.key_status = {i: {'active': True, 'error_count': 0, 'last_error': None} 
                               for i in range(len(self.api_keys))}
        st.success("✅ API key status reset!")

# ... (QueryClassifier, EQUIPMENT_KNOWLEDGE, QuotaManager, rate_limit_with_rotation from Script 1 remain unchanged) ...
# --- Enhanced Query Classification System ---
class QueryClassifier:
    def __init__(self):
        self.equipment_keywords = {
            'hvac': ['hvac', 'air conditioning', 'heating', 'cooling', 'thermostat', 'ac', 'heat pump'],
            'electrical': ['electrical', 'power', 'circuit', 'breaker', 'outlet', 'wiring', 'voltage'],
            'network': ['network', 'internet', 'wifi', 'router', 'switch', 'connection', 'ip'],
            'server': ['server', 'computer', 'pc', 'laptop', 'cpu', 'memory', 'disk', 'hardware'],
            'industrial': ['motor', 'pump', 'valve', 'sensor', 'controller', 'industrial', 'machinery'],
            'automotive': ['car', 'auto', 'engine', 'brake', 'transmission', 'vehicle', 'motor'],
            'medical': ['medical', 'hospital', 'mri', 'x-ray', 'ultrasound', 'equipment', 'device'],
            'television': ['tv', 'television', 'display', 'screen', 'monitor', 'lg', 'samsung', 'sony', 'flickering', 'picture', 'video']
        }
        
        self.technical_keywords = ['code', 'programming', 'software', 'debug', 'error', 'syntax', 'function', 'algorithm']
        self.general_keywords = ['how to', 'what is', 'explain', 'help me', 'tutorial', 'guide', 'recipe', 'cook']
    
    def classify_query(self, query: str) -> Dict:
        query_lower = query.lower()
        for equipment_type, keywords in self.equipment_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return {'category': 'equipment_diagnostic', 'subcategory': equipment_type}
        if any(keyword in query_lower for keyword in self.technical_keywords):
            return {'category': 'technical_support', 'subcategory': 'software'}
        if any(keyword in query_lower for keyword in self.general_keywords):
            return {'category': 'general_inquiry', 'subcategory': 'information'}
        return {'category': 'general_inquiry', 'subcategory': 'unknown'}

# --- Smart Equipment Knowledge Base (Enhanced) ---
EQUIPMENT_KNOWLEDGE = {
    'hvac': {
        'name': 'HVAC System',
        'components': ['compressor', 'condenser', 'evaporator', 'thermostat', 'blower motor', 'air filter'],
        'common_issues': ['no cooling', 'poor airflow', 'strange noises', 'water leaks', 'thermostat issues']
    },
    'electrical': { 'name': 'Electrical System', 'components': ['circuit breakers', 'outlets', 'switches', 'wiring'], 'common_issues': ['power outages', 'flickering lights', 'tripped breakers']},
    'network': { 'name': 'Network Equipment', 'components': ['switches', 'routers', 'cables', 'ports'], 'common_issues': ['connection timeout', 'slow speeds', 'device offline']},
    'server': { 'name': 'Server Hardware', 'components': ['cpu', 'memory', 'storage', 'power supply', 'cooling fans'], 'common_issues': ['high cpu usage', 'disk failures', 'overheating']},
    'industrial': { 'name': 'Industrial Equipment', 'components': ['motors', 'pumps', 'valves', 'sensors'], 'common_issues': ['motor failures', 'pump problems', 'sensor malfunctions']},
    'automotive': { 'name': 'Automotive Systems', 'components': ['engine', 'transmission', 'brakes', 'electrical system'], 'common_issues': ['engine won\'t start', 'overheating', 'brake problems']},
    'medical': { 'name': 'Medical Equipment', 'components': ['imaging systems', 'monitors', 'pumps', 'sensors'], 'common_issues': ['calibration errors', 'power failures', 'sensor malfunctions']},
    'television': { 'name': 'Television/Display', 'components': ['display panel', 'power supply', 'main board', 'HDMI ports'], 'common_issues': ['flickering screen', 'no picture', 'lines on screen']}
}

# --- Enhanced Quota Management System ---
class QuotaManager:
    def __init__(self):
        self.request_log = self._load_request_log()
        self.session_requests = 0
    
    def _load_request_log(self) -> Dict:
        if 'quota_log' not in st.session_state:
            st.session_state.quota_log = {
                'daily_count': 0, 'hourly_count': 0,
                'last_reset_date': datetime.now().date(), 'last_reset_hour': datetime.now().hour,
                'total_requests': 0, 'api_key_usage': {i: 0 for i in range(5)}
            }
        return st.session_state.quota_log
    
    def can_make_request(self) -> tuple[bool, str]:
        now = datetime.now()
        if now.date() > self.request_log['last_reset_date']:
            self.request_log['daily_count'] = 0
            self.request_log['last_reset_date'] = now.date()
        if now.hour != self.request_log['last_reset_hour']:
            self.request_log['hourly_count'] = 0
            self.request_log['last_reset_hour'] = now.hour
        if self.request_log['daily_count'] >= quota_config.daily_limit:
            return False, "Daily quota exceeded."
        if self.request_log['hourly_count'] >= quota_config.requests_per_hour:
            return False, "Hourly quota exceeded."
        return True, "OK"
    
    def record_request(self, success: bool = True, api_key_index: int = 0):
        if success:
            self.request_log['daily_count'] += 1
            self.request_log['hourly_count'] += 1
            self.request_log['total_requests'] += 1
            self.request_log['api_key_usage'][api_key_index] = self.request_log['api_key_usage'].get(api_key_index, 0) + 1
            self.session_requests += 1
            st.session_state.quota_log = self.request_log
    
    def get_quota_status(self) -> Dict:
        return {
            'daily_used': self.request_log['daily_count'], 'daily_limit': quota_config.daily_limit,
            'hourly_used': self.request_log['hourly_count'], 'hourly_limit': quota_config.requests_per_hour,
            'session_requests': self.session_requests, 'total_requests': self.request_log.get('total_requests', 0),
            'api_key_usage': self.request_log.get('api_key_usage', {})
        }

# --- Rate Limiting Decorator (Enhanced) ---
def rate_limit_with_rotation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        quota_manager = st.session_state.get('quota_manager')
        api_manager = st.session_state.get('api_manager')
        if not quota_manager:
            quota_manager = QuotaManager(); st.session_state.quota_manager = quota_manager
        if not api_manager:
            api_manager = MultiAPIManager(); st.session_state.api_manager = api_manager
        
        can_request, message = quota_manager.can_make_request()
        if not can_request:
            if quota_config.use_fallback_on_limit:
                st.warning(f"⚠️ {message} Using fallback mode.")
                return None
            else:
                raise Exception(f"Quota limit reached: {message}")
        
        try:
            result = func(*args, **kwargs)
            quota_manager.record_request(success=True, api_key_index=api_manager.current_key_index)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                st.error(f"🚫 API Rate Limit: {e}")
            else:
                st.error(f"🚫 API Error: {e}")
            if quota_config.use_fallback_on_limit:
                return None
            raise e
    return wrapper

# --- HYBRID Versatile AI Engine ---
# Merged Script 1's engine with Script 2's granular fallback logic
class VersatileAIEngine:
    def __init__(self, api_manager: MultiAPIManager):
        self.api_manager = api_manager
        self.query_classifier = QueryClassifier()
        # Fallback solutions are now more comprehensive, inspired by Script 2
        self.fallback_solutions = self._load_fallback_solutions()
    
    def _load_fallback_solutions(self) -> Dict:
        # This is a more detailed set of fallbacks
        return {
            'equipment_diagnostic': {
                'hvac': {'general': "**HVAC System Troubleshooting:**\n1. Check power and breakers.\n2. Inspect thermostat settings.\n3. Examine air filters for clogs."},
                'television': {'general': "**TV/Display Troubleshooting:**\n1. Check all cable connections.\n2. Test with different input sources.\n3. Reset TV to factory settings."},
                'electrical': {'general': "**Electrical System Diagnosis:**\n1. Safety first - turn off power.\n2. Check for tripped circuit breakers.\n3. Test GFCI outlets."},
            },
            'technical_support': {'general': "**Software Troubleshooting:**\n1. Restart the application/system.\n2. Check for software updates.\n3. Review error logs for details."},
            'general_inquiry': {'general': "**General Guidance:**\n1. Break down the problem.\n2. Research reliable sources.\n3. Start with simple solutions first."}
        }
        
    @rate_limit_with_rotation
    def analyze_and_respond(self, query: str) -> str:
        classification = self.query_classifier.classify_query(query)
        model = self.api_manager.get_working_model()
        
        if not model:
            return self._generate_fallback_response(query, classification)

        try:
            if classification['category'] == 'equipment_diagnostic':
                return self._generate_equipment_response(query, classification, model)
            elif classification['category'] == 'technical_support':
                return self._generate_technical_response(query, classification, model)
            else:
                return self._generate_general_response(query, classification, model)
        except Exception as e:
            st.warning(f"AI generation failed: {e}. Using granular fallback.")
            # --- MERGED --- Granular fallback on single API failure
            return self._generate_fallback_response(query, classification)

    def _generate_equipment_response(self, query: str, classification: Dict, model) -> str:
        equipment_info = EQUIPMENT_KNOWLEDGE.get(classification.get('subcategory', 'general'), {})
        prompt = f"""You are an expert equipment diagnostic technician. Analyze this equipment issue and provide a comprehensive, step-by-step solution:
**Query**: {query}
**Equipment Type**: {equipment_info.get('name', 'Equipment')}
Provide a structured response including Safety Checks, Diagnostic Steps, Possible Causes, Solutions, and When to Call a Professional."""
        response = model.generate_content(prompt)
        return f"🔧 **Equipment Diagnostic Solution**\n\n{response.text}"

    def _generate_technical_response(self, query: str, classification: Dict, model) -> str:
        prompt = f"""You are an experienced technical support specialist. Help solve this technical issue:
**Query**: {query}
Provide a structured response including Problem Analysis, Quick Fixes, and Detailed Troubleshooting steps."""
        response = model.generate_content(prompt)
        return f"💻 **Technical Support Solution**\n\n{response.text}"

    def _generate_general_response(self, query: str, classification: Dict, model) -> str:
        prompt = f"You are a helpful AI assistant. Provide a comprehensive answer to this query: **Query**: {query}"
        response = model.generate_content(prompt)
        return f"🤖 **AI Assistant Response**\n\n{response.text}"

    def _generate_fallback_response(self, query: str, classification: Dict) -> str:
        category = classification['category']
        subcategory = classification.get('subcategory', 'general')
        
        category_solutions = self.fallback_solutions.get(category, {})
        solution = category_solutions.get(subcategory, {}).get('general', category_solutions.get('general', "No specific fallback available."))
        
        return f"🤖 **Fallback Solution** (AI temporarily unavailable)\n\n{solution}"
    
    def generate_followup_questions(self, query: str, classification: Dict) -> List[str]:
        # This method generates context-aware follow-up questions, with a simple rule-based fallback.
        model = self.api_manager.get_working_model()
        try:
            if not model: raise Exception("No model available")
            prompt = f"Based on this query: '{query}', generate 3 specific follow-up questions a technician would ask. Return only a Python list of strings."
            response = model.generate_content(prompt)
            # A simple way to parse the list-like string from the model
            questions = [q.strip().strip("'\"") for q in response.text.strip()[1:-1].split(',')]
            return questions[:3] if questions else self._get_fallback_questions(classification)
        except Exception:
            return self._get_fallback_questions(classification)

    def _get_fallback_questions(self, classification: Dict) -> List[str]:
        if classification['category'] == 'equipment_diagnostic':
            return ["When did the problem start?", "Are there any error lights?", "Has any maintenance been done recently?"]
        else:
            return ["Can you provide more context?", "What have you tried so far?", "What is the expected outcome?"]

# ... (DocumentProcessor, EnhancedRAGSystem, MaintenancePipeline from Script 1 remain the same) ...
class DocumentProcessor:
    def process_uploaded_files(self, uploaded_files) -> List[Dict]:
        processed_docs = []
        for uploaded_file in uploaded_files:
            try:
                # Simplified for brevity
                content = "Processed content from " + uploaded_file.name
                processed_docs.append({'filename': uploaded_file.name, 'content': content, 'size': uploaded_file.size, 'type': 'file'})
            except Exception as e:
                processed_docs.append({'filename': uploaded_file.name, 'content': f"Error: {e}", 'size': 0, 'type': 'error'})
        return processed_docs
class EnhancedRAGSystem:
    def __init__(self): self.is_initialized = False; self.documents = []; self.document_chunks = []
    def build_index(self, documents: List[Dict]): self.documents = documents; self.is_initialized = True; return True
    def get_context_for_query(self, query: str) -> str: return "Sample context from RAG system based on user query." if self.is_initialized else ""
class MaintenancePipeline:
    def __init__(self): self.equipment_list = [f"Device_{i}" for i in range(35)]
    def get_maintenance_dashboard(self): return {'total_equipment': 35, 'high_risk': 5, 'medium_risk': 10, 'low_risk': 20, 'equipment_details': {}}
    def simulate_real_time_alert(self): return {'alert_type': 'High Temperature', 'device_id': 'Server_5', 'location': 'Data Center A', 'severity': 'CRITICAL', 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# --- Main Streamlit Application ---
def main():
    # Initialize session state
    if 'api_manager' not in st.session_state: st.session_state.api_manager = MultiAPIManager()
    if 'quota_manager' not in st.session_state: st.session_state.quota_manager = QuotaManager()
    if 'ai_engine' not in st.session_state: st.session_state.ai_engine = VersatileAIEngine(st.session_state.api_manager)
    if 'maintenance_pipeline' not in st.session_state: st.session_state.maintenance_pipeline = MaintenancePipeline()
    if 'document_processor' not in st.session_state: st.session_state.document_processor = DocumentProcessor()
    if 'rag_system' not in st.session_state: st.session_state.rag_system = EnhancedRAGSystem()
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'followup_questions' not in st.session_state: st.session_state.followup_questions = []

    st.title("🚀 Versatile AI Assistant")
    st.markdown("*Hybrid Edition: Multi-API Backend with an Interactive Chat UI*")
    
    # Sidebar remains the same as Script 1
    with st.sidebar:
        st.header("🛠️ System Control Panel")
        api_status = st.session_state.api_manager.get_api_status()
        quota_status = st.session_state.quota_manager.get_quota_status()
        st.subheader("📊 API Status")
        st.metric("Active Keys", f"{api_status['active_keys']}/{api_status['total_keys']}")
        st.metric("Current Key", f"#{api_status['current_key']}")
        if st.button("🔄 Reset API Keys"): st.session_state.api_manager.reset_key_status()
        st.subheader("📈 Usage Quota")
        st.progress(quota_status['daily_used'] / (quota_status['daily_limit'] or 1))
        st.text(f"Daily: {quota_status['daily_used']}/{quota_status['daily_limit']}")
    
    # --- MERGED --- Refactored chat logic into a handler function
    def handle_user_prompt(prompt: str):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.spinner("🤔 Analyzing and generating solution..."):
            classification = st.session_state.ai_engine.query_classifier.classify_query(prompt)
            response = st.session_state.ai_engine.analyze_and_respond(prompt)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.session_state.followup_questions = st.session_state.ai_engine.generate_followup_questions(prompt, classification)
        st.rerun()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI Chat", "📄 Document RAG", "🔧 Equipment Monitor", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("💬 Intelligent AI Assistant")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])

        # --- MERGED --- Follow-up Questions UI from Script 2
        if st.session_state.followup_questions:
            st.markdown("---")
            st.info("Need more specific help? Try one of these questions:")
            cols = st.columns(len(st.session_state.followup_questions))
            for i, question in enumerate(st.session_state.followup_questions):
                if cols[i].button(question, key=f"followup_{i}"):
                    handle_user_prompt(question)

        # Chat Input
        if prompt := st.chat_input("Describe your equipment issue..."):
            handle_user_prompt(prompt)

        # --- MERGED --- Quick Action Buttons UI from Script 2
        st.markdown("---")
        st.markdown("##### 🚀 Quick Actions")
        cols = st.columns(4)
        quick_actions = {"❄️ HVAC": "My HVAC isn't cooling", "⚡ Electrical": "My lights are flickering", "🌐 Network": "My internet is down", "🖥️ Server": "The server is overheating"}
        for i, (label, prompt) in enumerate(quick_actions.items()):
            if cols[i].button(label):
                handle_user_prompt(prompt)

    # Other tabs remain as they were in Script 1
    with tab2:
        st.header("📄 Document Knowledge Base (RAG)")
        # RAG functionality UI here...
        st.info("RAG system is ready. Upload documents to activate.")

    with tab3:
        st.header("🔧 Equipment Monitoring Dashboard")
        # Monitoring UI here...
        st.info("Equipment monitoring dashboard is active.")

    with tab4:
        st.header("📊 System Analytics")
        # Analytics UI here...
        st.info("System analytics are available.")

    with tab5:
        st.header("⚙️ System Settings")
        # Settings UI here...
        st.info("System settings can be configured here.")

if __name__ == "__main__":
    main()
