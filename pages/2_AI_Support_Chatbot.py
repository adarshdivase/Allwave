# app.py - Enhanced Versatile AI Assistant with Improved CSS Styling
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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #2563eb;
        --primary-hover: #1d4ed8;
        --secondary-color: #f8fafc;
        --accent-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --success-color: #22c55e;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-success: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        --gradient-warning: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        --gradient-error: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    
    /* Dark mode support */
    [data-theme="dark"] {
        --primary-color: #3b82f6;
        --secondary-color: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border-color: #374151;
    }
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: var(--gradient-primary);
        color: white;
        padding: 2rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Custom card components */
    .custom-card {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        border-radius: var(--radius-md);
        margin: 0.5rem 0;
    }
    
    .status-success {
        border-left-color: var(--success-color);
        background: var(--gradient-success);
        color: #065f46;
    }
    
    .status-warning {
        border-left-color: var(--warning-color);
        background: var(--gradient-warning);
        color: #92400e;
    }
    
    .status-error {
        border-left-color: var(--error-color);
        background: var(--gradient-error);
        color: #991b1b;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--gradient-primary);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Alert components */
    .alert {
        padding: 1rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }
    
    .alert-critical {
        background: #fef2f2;
        border-color: #fecaca;
        color: #991b1b;
    }
    
    .alert-high {
        background: #fffbeb;
        border-color: #fed7aa;
        color: #92400e;
    }
    
    .alert-medium {
        background: #fefce8;
        border-color: #fde047;
        color: #a16207;
    }
    
    .alert-low {
        background: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    
    /* Button enhancements */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar-header {
        background: var(--gradient-primary);
        color: white;
        padding: 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--secondary-color);
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
        border-color: var(--primary-color);
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: var(--radius-md);
        border: 2px solid var(--border-color);
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary);
        border-radius: var(--radius-sm);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: var(--radius-lg);
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    /* Chat message styling */
    .chat-message {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        position: relative;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: var(--gradient-primary);
        border-radius: var(--radius-sm);
    }
    
    .chat-query {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
    }
    
    .chat-response {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    .chat-timestamp {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--border-color);
    }
    
    /* Equipment dashboard */
    .equipment-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .equipment-item {
        background: white;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .equipment-item:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Footer styling */
    .custom-footer {
        background: var(--secondary-color);
        padding: 2rem;
        border-radius: var(--radius-lg);
        margin-top: 3rem;
        text-align: center;
        border: 1px solid var(--border-color);
    }
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--border-color);
        border-radius: 50%;
        border-top-color: var(--primary-color);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .main-header h1 {
            font-size: 2rem;
        }
        
        .equipment-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--secondary-color);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: var(--radius-sm);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-hover);
    }
    </style>
    """, unsafe_allow_html=True)

# Custom component functions for enhanced UI
def create_custom_card(title, content, card_type="default"):
    """Create a custom styled card component"""
    type_classes = {
        "success": "status-success",
        "warning": "status-warning", 
        "error": "status-error",
        "default": ""
    }
    
    st.markdown(f"""
    <div class="custom-card {type_classes.get(card_type, '')} fade-in">
        <div class="card-header">{title}</div>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(label, value, delta=None):
    """Create a custom metric card"""
    delta_html = ""
    if delta is not None:
        delta_color = "var(--success-color)" if delta >= 0 else "var(--error-color)"
        delta_symbol = "+" if delta >= 0 else ""
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem; margin-top: 0.25rem;">{delta_symbol}{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card fade-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_alert_box(message, alert_type="info", title=None):
    """Create a custom alert box"""
    type_icons = {
        "critical": "🔴",
        "high": "🟠", 
        "medium": "🟡",
        "low": "🟢",
        "info": "ℹ️"
    }
    
    title_html = f"<strong>{title}</strong><br>" if title else ""
    
    st.markdown(f"""
    <div class="alert alert-{alert_type} fade-in">
        <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
            <span style="font-size: 1.2rem;">{type_icons.get(alert_type, "ℹ️")}</span>
            <div>{title_html}{message}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_status_indicator(status, label):
    """Create a status indicator component"""
    status_colors = {
        "online": "#22c55e",
        "offline": "#ef4444", 
        "warning": "#f59e0b",
        "maintenance": "#8b5cf6"
    }
    
    color = status_colors.get(status.lower(), "#64748b")
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0;">
        <div style="width: 10px; height: 10px; border-radius: 50%; background: {color};" class="pulse"></div>
        <span style="font-weight: 500; color: var(--text-primary);">{label}</span>
    </div>
    """, unsafe_allow_html=True)

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

# --- Multi-API Key Management System ---
class MultiAPIManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.key_status = {i: {'active': True, 'error_count': 0, 'last_error': None} 
                           for i in range(len(self.api_keys))}
        self.current_model = None
        self._initialize_current_model()
    
    def _load_api_keys(self) -> List[str]:
        """Load all available API keys from secrets"""
        keys = []
        for i in range(1, 6):  # Check for GEMINI_API_KEY_1 through GEMINI_API_KEY_5
            key_name = f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY"
            if key_name in st.secrets:
                keys.append(st.secrets[key_name])
        
        if not keys:
            st.error("❌ No API keys found. Please add GEMINI_API_KEY_1 through GEMINI_API_KEY_5 to secrets.")
        
        return keys
    
    def _initialize_current_model(self):
        """Initialize the current model with the active API key"""
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
        """Get a working model, rotating through API keys if needed"""
        max_attempts = len(self.api_keys)
        
        for attempt in range(max_attempts):
            if self.key_status[self.current_key_index]['active']:
                try:
                    if not self.current_model:
                        self._initialize_current_model()
                    
                    # Test the current model with a simple request
                    test_response = self.current_model.generate_content("Test")
                    
                    # Reset error count on successful test
                    self.key_status[self.current_key_index]['error_count'] = 0
                    return self.current_model
                    
                except Exception as e:
                    self._handle_api_error(e)
            
            # Move to next API key
            self._rotate_to_next_key()
        
        # All keys failed
        st.error("🚫 All API keys exhausted. Using fallback mode.")
        return None
    
    def _handle_api_error(self, error: Exception):
        """Handle API errors and update key status"""
        error_str = str(error).lower()
        current_status = self.key_status[self.current_key_index]
        
        current_status['error_count'] += 1
        current_status['last_error'] = str(error)
        
        # Disable key if too many errors or quota exceeded
        if (current_status['error_count'] >= 3 or 
            'quota' in error_str or 
            '429' in error_str or 
            'rate limit' in error_str):
            
            current_status['active'] = False
            st.warning(f"⚠️ API Key {self.current_key_index + 1} disabled due to: {error}")
    
    def _rotate_to_next_key(self):
        """Rotate to the next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.current_model = None
        self._initialize_current_model()
    
    def get_api_status(self) -> Dict:
        """Get current API key status"""
        active_keys = sum(1 for status in self.key_status.values() if status['active'])
        return {
            'total_keys': len(self.api_keys),
            'active_keys': active_keys,
            'current_key': self.current_key_index + 1,
            'key_status': self.key_status
        }
    
    def reset_key_status(self, key_index: int = None):
        """Reset error status for a specific key or all keys"""
        if key_index is not None:
            self.key_status[key_index] = {'active': True, 'error_count': 0, 'last_error': None}
        else:
            self.key_status = {i: {'active': True, 'error_count': 0, 'last_error': None} 
                               for i in range(len(self.api_keys))}
        st.success("✅ API key status reset!")

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
        """Classify query into equipment, technical, or general category"""
        query_lower = query.lower()
        
        # Check for equipment-related queries
        for equipment_type, keywords in self.equipment_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    'category': 'equipment_diagnostic',
                    'subcategory': equipment_type,
                    'confidence': 0.9,
                    'keywords': [kw for kw in keywords if kw in query_lower]
                }
        
        # Check for technical support queries
        if any(keyword in query_lower for keyword in self.technical_keywords):
            return {
                'category': 'technical_support',
                'subcategory': 'software',
                'confidence': 0.8,
                'keywords': [kw for kw in self.technical_keywords if kw in query_lower]
            }
        
        # Check for general queries
        if any(keyword in query_lower for keyword in self.general_keywords):
            return {
                'category': 'general_inquiry',
                'subcategory': 'information',
                'confidence': 0.7,
                'keywords': [kw for kw in self.general_keywords if kw in query_lower]
            }
        
        # Default classification
        return {
            'category': 'general_inquiry',
            'subcategory': 'unknown',
            'confidence': 0.5,
            'keywords': []
        }

# [Include all other classes from the original code - QuotaManager, VersatileAIEngine, MaintenancePipeline, DocumentProcessor, EnhancedRAGSystem]
# For brevity, I'm showing the main() function with enhanced styling

# --- Main Streamlit Application with Enhanced Styling ---
def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'api_manager' not in st.session_state:
        st.session_state.api_manager = MultiAPIManager()
    
    if 'quota_manager' not in st.session_state:
        st.session_state.quota_manager = QuotaManager()
    
    # App Header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Versatile AI Assistant</h1>
        <p>Advanced Multi-Modal AI with Equipment Diagnostics, RAG, and Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3 style="margin: 0; font-weight: 600;">🛠️ Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # API Status with custom components
        api_status = st.session_state.api_manager.get_api_status()
        quota_status = st.session_state.quota_manager.get_quota_status()
        
        st.markdown("### 📊 API Status")
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Active Keys", f"{api_status['active_keys']}/{api_status['total_keys']}")
        with col2:
            create_metric_card("Current Key", f"#{api_status['current_key']}")
        
        # Status indicators
        for i, status in api_status['key_status'].items():
            status_type = "online" if status['active'] else "offline"
            create_status_indicator(status_type, f"API Key {i+1}")
        
        if st.button("🔄 Reset API Keys", key="reset_keys"):
            st.session_state.api_manager.reset_key_status()
            st.rerun()
        
        st.markdown("### 📈 Usage Quota")
        progress_value = quota_status['daily_used'] / max(quota_status['daily_limit'], 1)
        st.progress(progress_value)
        
        create_custom_card("Daily Usage", 
                          f"{quota_status['daily_used']}/{quota_status['daily_limit']} requests")
        create_custom_card("Hourly Usage", 
                          f"{quota_status['hourly_used']}/{quota_status['hourly_limit']} requests")
        create_custom_card("Session Requests", 
                          f"{quota_status['session_requests']} total")
    
    # Main Tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat", "📄 Document RAG", "🔧 Equipment Monitor", "📊 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        st.markdown("## 💬 Intelligent AI Assistant")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Query input with enhanced styling
        user_query = st.text_area("✨ Ask me anything:", height=120, 
                                  placeholder="Example: My TV is flickering, what should I check?")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🚀 Ask AI", type="primary", key="ask_ai"):
                if user_query:
                    with st.spinner("🤔 Analyzing your query..."):
         # --- Remaining classes that were referenced but not shown ---

class QuotaManager:
    """Enhanced quota management with per-key tracking"""
    
    def __init__(self):
        self.daily_limit = quota_config.daily_limit
        self.hourly_limit = quota_config.requests_per_hour
        self.reset_quota_if_needed()
        
        # Initialize session counters
        if 'session_requests' not in st.session_state:
            st.session_state.session_requests = 0
    
    def reset_quota_if_needed(self):
        """Reset quota counters if day has changed"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        
        # Daily reset
        if 'quota_date' not in st.session_state or st.session_state.quota_date != today:
            st.session_state.quota_date = today
            st.session_state.daily_requests = 0
        
        # Hourly reset
        if 'quota_hour' not in st.session_state or st.session_state.quota_hour != current_hour:
            st.session_state.quota_hour = current_hour
            st.session_state.hourly_requests = 0
    
    def can_make_request(self) -> Tuple[bool, str]:
        """Check if request can be made within quota limits"""
        self.reset_quota_if_needed()
        
        daily_used = getattr(st.session_state, 'daily_requests', 0)
        hourly_used = getattr(st.session_state, 'hourly_requests', 0)
        
        if daily_used >= self.daily_limit:
            return False, f"Daily limit exceeded ({daily_used}/{self.daily_limit})"
        
        if hourly_used >= self.hourly_limit:
            return False, f"Hourly limit exceeded ({hourly_used}/{self.hourly_limit})"
        
        return True, "OK"
    
    def record_request(self):
        """Record a successful request"""
        st.session_state.daily_requests = getattr(st.session_state, 'daily_requests', 0) + 1
        st.session_state.hourly_requests = getattr(st.session_state, 'hourly_requests', 0) + 1
        st.session_state.session_requests += 1
    
    def get_quota_status(self) -> Dict:
        """Get current quota status"""
        self.reset_quota_if_needed()
        return {
            'daily_limit': self.daily_limit,
            'daily_used': getattr(st.session_state, 'daily_requests', 0),
            'hourly_limit': self.hourly_limit,
            'hourly_used': getattr(st.session_state, 'hourly_requests', 0),
            'session_requests': st.session_state.session_requests
        }

class VersatileAIEngine:
    """Enhanced AI engine with multi-modal capabilities"""
    
    def __init__(self, api_manager: MultiAPIManager):
        self.api_manager = api_manager
        self.classifier = QueryClassifier()
        self.conversation_memory = []
        self.max_memory = 10
    
    def process_query(self, query: str, context: str = "", image_data=None) -> Dict:
        """Process query with enhanced context awareness"""
        # Classify the query
        classification = self.classifier.classify_query(query)
        
        # Get working model
        model = self.api_manager.get_working_model()
        if not model:
            return self._get_fallback_response(query, classification)
        
        try:
            # Build enhanced prompt based on classification
            enhanced_prompt = self._build_enhanced_prompt(query, classification, context)
            
            # Handle multi-modal input
            if image_data:
                response = model.generate_content([enhanced_prompt, image_data])
            else:
                response = model.generate_content(enhanced_prompt)
            
            # Process and structure response
            processed_response = self._process_response(response.text, classification)
            
            # Update conversation memory
            self._update_memory(query, processed_response['content'])
            
            return {
                'success': True,
                'content': processed_response['content'],
                'classification': classification,
                'confidence': processed_response['confidence'],
                'sources': processed_response.get('sources', []),
                'follow_up_questions': processed_response.get('follow_up', []),
                'api_key_used': self.api_manager.current_key_index + 1
            }
            
        except Exception as e:
            self.api_manager._handle_api_error(e)
            return self._get_fallback_response(query, classification, str(e))
    
    def _build_enhanced_prompt(self, query: str, classification: Dict, context: str) -> str:
        """Build context-aware prompt based on query classification"""
        base_prompt = f"""
        You are a versatile AI assistant specializing in equipment diagnostics, technical support, and general inquiries.
        
        Query Classification: {classification['category']} ({classification['subcategory']})
        User Query: {query}
        """
        
        if classification['category'] == 'equipment_diagnostic':
            base_prompt += """
            
            Equipment Diagnostic Mode:
            - Provide step-by-step troubleshooting steps
            - Include safety warnings where applicable
            - Suggest when professional help is needed
            - List common causes and solutions
            - Include maintenance recommendations
            """
        
        elif classification['category'] == 'technical_support':
            base_prompt += """
            
            Technical Support Mode:
            - Provide clear, actionable solutions
            - Include code examples if relevant
            - Explain technical concepts simply
            - Offer alternative approaches
            """
        
        if context:
            base_prompt += f"\n\nAdditional Context: {context}"
        
        # Add conversation memory for context
        if self.conversation_memory:
            memory_context = "\n".join([f"Q: {item['query']}\nA: {item['response'][:200]}..." 
                                       for item in self.conversation_memory[-3:]])
            base_prompt += f"\n\nConversation History:\n{memory_context}"
        
        return base_prompt
    
    def _process_response(self, response_text: str, classification: Dict) -> Dict:
        """Process and enhance AI response based on classification"""
        # Base processing
        processed = {
            'content': response_text,
            'confidence': classification['confidence'],
            'sources': [],
            'follow_up': []
        }
        
        # Add follow-up questions based on category
        if classification['category'] == 'equipment_diagnostic':
            processed['follow_up'] = [
                "What specific symptoms are you experiencing?",
                "When did this issue first occur?",
                "Have you tried any troubleshooting steps already?",
                "Is this affecting other equipment as well?"
            ]
        
        elif classification['category'] == 'technical_support':
            processed['follow_up'] = [
                "What programming language are you using?",
                "Can you share the error message?",
                "What have you tried so far?",
                "Would you like code examples?"
            ]
        
        return processed
    
    def _update_memory(self, query: str, response: str):
        """Update conversation memory"""
        self.conversation_memory.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now()
        })
        
        # Keep only recent conversations
        if len(self.conversation_memory) > self.max_memory:
            self.conversation_memory.pop(0)
    
    def _get_fallback_response(self, query: str, classification: Dict, error: str = "") -> Dict:
        """Provide fallback response when API is unavailable"""
        fallback_responses = {
            'equipment_diagnostic': "I'm currently unable to access my diagnostic database. For equipment issues, please check: 1) Power connections, 2) Error displays, 3) Recent changes, 4) Basic troubleshooting steps in your manual.",
            'technical_support': "I'm currently in offline mode. For technical issues, please check documentation, community forums, or try breaking down the problem into smaller parts.",
            'general_inquiry': "I'm currently unable to provide detailed responses. Please try again later or consult relevant documentation."
        }
        
        return {
            'success': False,
            'content': fallback_responses.get(classification['category'], fallback_responses['general_inquiry']),
            'classification': classification,
            'confidence': 0.3,
            'error': error,
            'fallback': True
        }

class DocumentProcessor:
    """Enhanced document processing with multi-format support"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.md', '.docx', '.csv']
        self.chunk_size = config.chunk_size
        self.overlap = 50
    
    def process_document(self, file_path: str) -> List[Dict]:
        """Process document and return structured chunks"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return self._process_pdf(file_path)
            elif file_ext in ['.txt', '.md']:
                return self._process_text(file_path)
            elif file_ext == '.docx':
                return self._process_docx(file_path)
            elif file_ext == '.csv':
                return self._process_csv(file_path)
            else:
                st.error(f"Unsupported file format: {file_ext}")
                return []
                
        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: str) -> List[Dict]:
        """Process PDF files using PyMuPDF"""
        chunks = []
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split into chunks
            page_chunks = self._create_chunks(text, page_num + 1)
            chunks.extend(page_chunks)
        
        doc.close()
        return chunks
    
    def _process_text(self, file_path: str) -> List[Dict]:
        """Process text and markdown files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self._create_chunks(text, source=os.path.basename(file_path))
    
    def _process_docx(self, file_path: str) -> List[Dict]:
        """Process DOCX files"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return self._create_chunks(text, source=os.path.basename(file_path))
        except ImportError:
            st.error("python-docx package required for DOCX processing")
            return []
    
    def _process_csv(self, file_path: str) -> List[Dict]:
        """Process CSV files"""
        df = pd.read_csv(file_path)
        
        # Convert CSV to text representation
        text = f"Dataset: {os.path.basename(file_path)}\n"
        text += f"Columns: {', '.join(df.columns)}\n"
        text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        
        # Add summary statistics
        text += "Summary:\n" + str(df.describe()) + "\n\n"
        
        # Add sample data
        text += "Sample Data:\n" + df.head().to_string()
        
        return self._create_chunks(text, source=os.path.basename(file_path))
    
    def _create_chunks(self, text: str, source: any = None) -> List[Dict]:
        """Create overlapping chunks from text"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                'content': chunk_text,
                'source': source,
                'chunk_id': len(chunks),
                'word_count': len(chunk_words),
                'timestamp': datetime.now()
            })
        
        return chunks

class EnhancedRAGSystem:
    """Enhanced RAG system with improved retrieval and generation"""
    
    def __init__(self):
        self.encoder = None
        self.index = None
        self.documents = []
        self.doc_processor = DocumentProcessor()
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize the sentence transformer model"""
        try:
            with st.spinner("Loading embedding model..."):
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            st.success("✅ Embedding model loaded successfully")
        except Exception as e:
            st.error(f"❌ Failed to load embedding model: {e}")
    
    def add_documents(self, file_paths: List[str]) -> Dict:
        """Add multiple documents to the RAG system"""
        results = {'success': 0, 'failed': 0, 'total_chunks': 0}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_path in enumerate(file_paths):
            try:
                status_text.text(f"Processing: {os.path.basename(file_path)}")
                
                # Process document
                chunks = self.doc_processor.process_document(file_path)
                
                if chunks:
                    self.documents.extend(chunks)
                    results['success'] += 1
                    results['total_chunks'] += len(chunks)
                    
                    st.success(f"✅ Added {len(chunks)} chunks from {os.path.basename(file_path)}")
                else:
                    results['failed'] += 1
                    st.warning(f"⚠️ No content extracted from {os.path.basename(file_path)}")
                
            except Exception as e:
                results['failed'] += 1
                st.error(f"❌ Failed to process {os.path.basename(file_path)}: {e}")
            
            progress_bar.progress((i + 1) / len(file_paths))
        
        # Build FAISS index
        if self.documents and self.encoder:
            self._build_index()
        
        status_text.empty()
        progress_bar.empty()
        
        return results
    
    def _build_index(self):
        """Build FAISS index for efficient similarity search"""
        try:
            with st.spinner("Building search index..."):
                # Extract content for embedding
                texts = [doc['content'] for doc in self.documents]
                
                # Generate embeddings
                embeddings = self.encoder.encode(texts)
                
                # Build FAISS index
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings.astype('float32'))
                
            st.success(f"✅ Search index built with {len(texts)} documents")
            
        except Exception as e:
            st.error(f"❌ Failed to build search index: {e}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents"""
        if not self.encoder or not self.index or not self.documents:
            return []
        
        if top_k is None:
            top_k = config.top_k_retrieval
        
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= config.similarity_threshold:
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            
            return results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def generate_answer(self, query: str, ai_engine: VersatileAIEngine) -> Dict:
        """Generate answer using RAG approach"""
        # Search for relevant documents
        relevant_docs = self.search(query)
        
        if not relevant_docs:
            return ai_engine.process_query(query, "No relevant documents found in knowledge base.")
        
        # Build context from relevant documents
        context = "Relevant Information:\n\n"
        for i, doc in enumerate(relevant_docs[:3], 1):
            context += f"{i}. {doc['content'][:500]}...\n\n"
        
        context += f"\nBased on the above information from {len(relevant_docs)} relevant sources, please answer the following question:"
        
        # Generate answer with context
        response = ai_engine.process_query(query, context)
        response['rag_sources'] = relevant_docs
        response['source_count'] = len(relevant_docs)
        
        return response

class MaintenancePipeline:
    """Equipment maintenance and monitoring pipeline"""
    
    def __init__(self):
        self.equipment_db = self._load_equipment_data()
        self.maintenance_schedules = self._load_maintenance_schedules()
        self.alerts = []
    
    def _load_equipment_data(self) -> List[Dict]:
        """Load or generate sample equipment data"""
        # Sample equipment data - in real scenario, load from database
        return [
            {
                'id': 'HVAC-001',
                'name': 'Main HVAC Unit',
                'type': 'hvac',
                'location': 'Building A - Roof',
                'status': 'online',
                'last_maintenance': '2024-01-15',
                'next_maintenance': '2024-04-15',
                'alerts': 2,
                'efficiency': 87,
                'runtime_hours': 2340
            },
            {
                'id': 'GEN-001', 
                'name': 'Backup Generator',
                'type': 'electrical',
                'location': 'Building A - Basement',
                'status': 'standby',
                'last_maintenance': '2024-02-01',
                'next_maintenance': '2024-05-01',
                'alerts': 0,
                'efficiency': 92,
                'runtime_hours': 156
            },
            {
                'id': 'NET-001',
                'name': 'Main Network Switch',
                'type': 'network',
                'location': 'Server Room',
                'status': 'online',
                'last_maintenance': '2024-01-30',
                'next_maintenance': '2024-07-30',
                'alerts': 1,
                'efficiency': 95,
                'runtime_hours': 8760
            }
        ]
    
    def _load_maintenance_schedules(self) -> List[Dict]:
        """Load maintenance schedules"""
        return [
            {
                'equipment_id': 'HVAC-001',
                'task': 'Filter Replacement',
                'frequency': 'Monthly',
                'last_completed': '2024-02-01',
                'next_due': '2024-03-01',
                'priority': 'high'
            },
            {
                'equipment_id': 'HVAC-001',
                'task': 'Coil Cleaning',
                'frequency': 'Quarterly',
                'last_completed': '2024-01-15',
                'next_due': '2024-04-15',
                'priority': 'medium'
            }
        ]
    
    def get_equipment_dashboard(self) -> Dict:
        """Generate equipment dashboard data"""
        total_equipment = len(self.equipment_db)
        online_count = sum(1 for eq in self.equipment_db if eq['status'] == 'online')
        offline_count = sum(1 for eq in self.equipment_db if eq['status'] == 'offline')
        warning_count = sum(1 for eq in self.equipment_db if eq['alerts'] > 0)
        
        return {
            'total_equipment': total_equipment,
            'online': online_count,
            'offline': offline_count,
            'warnings': warning_count,
            'avg_efficiency': sum(eq['efficiency'] for eq in self.equipment_db) / total_equipment,
            'equipment_list': self.equipment_db,
            'maintenance_due': self._get_maintenance_due()
        }
    
    def _get_maintenance_due(self) -> List[Dict]:
        """Get equipment due for maintenance"""
        due_items = []
        current_date = datetime.now().date()
        
        for schedule in self.maintenance_schedules:
            due_date = datetime.strptime(schedule['next_due'], '%Y-%m-%d').date()
            days_until_due = (due_date - current_date).days
            
            if days_until_due <= 7:  # Due within a week
                due_items.append({
                    **schedule,
                    'days_until_due': days_until_due,
                    'urgency': 'critical' if days_until_due <= 0 else 'high' if days_until_due <= 3 else 'medium'
                })
        
        return sorted(due_items, key=lambda x: x['days_until_due'])
    
    def generate_equipment_report(self, equipment_id: str = None) -> str:
        """Generate detailed equipment report"""
        if equipment_id:
            equipment = next((eq for eq in self.equipment_db if eq['id'] == equipment_id), None)
            if not equipment:
                return "Equipment not found"
            
            report = f"""
            # Equipment Report: {equipment['name']}
            
            **Equipment ID:** {equipment['id']}
            **Type:** {equipment['type'].title()}
            **Location:** {equipment['location']}
            **Status:** {equipment['status'].title()}
            
            ## Performance Metrics
            - **Efficiency:** {equipment['efficiency']}%
            - **Runtime Hours:** {equipment['runtime_hours']:,}
            - **Active Alerts:** {equipment['alerts']}
            
            ## Maintenance Information
            - **Last Maintenance:** {equipment['last_maintenance']}
            - **Next Maintenance:** {equipment['next_maintenance']}
            
            ## Recommendations
            """
            
            # Add recommendations based on status
            if equipment['efficiency'] < 80:
                report += "- ⚠️ **Low Efficiency Alert**: Consider immediate maintenance\n"
            
            if equipment['alerts'] > 0:
                report += f"- 🚨 **Active Alerts**: {equipment['alerts']} issues require attention\n"
            
            return report
        else:
            # Generate summary report for all equipment
            dashboard = self.get_equipment_dashboard()
            return f"""
            # Equipment Fleet Summary Report
            
            ## Overview
            - **Total Equipment:** {dashboard['total_equipment']}
            - **Online:** {dashboard['online']} ({dashboard['online']/dashboard['total_equipment']*100:.1f}%)
            - **Offline:** {dashboard['offline']}
            - **With Warnings:** {dashboard['warnings']}
            - **Average Efficiency:** {dashboard['avg_efficiency']:.1f}%
            
            ## Maintenance Due
            {len(dashboard['maintenance_due'])} items due for maintenance within 7 days
            """

# Continue with the main() function completion
def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'api_manager' not in st.session_state:
        st.session_state.api_manager = MultiAPIManager()
    
    if 'quota_manager' not in st.session_state:
        st.session_state.quota_manager = QuotaManager()
    
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = VersatileAIEngine(st.session_state.api_manager)
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedRAGSystem()
    
    if 'maintenance_pipeline' not in st.session_state:
        st.session_state.maintenance_pipeline = MaintenancePipeline()
    
    # App Header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Versatile AI Assistant</h1>
        <p>Advanced Multi-Modal AI with Equipment Diagnostics, RAG, and Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3 style="margin: 0; font-weight: 600;">🛠️ Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # API Status with custom components
        api_status = st.session_state.api_manager.get_api_status()
        quota_status = st.session_state.quota_manager.get_quota_status()
        
        st.markdown("### 📊 API Status")
        
        col1, col2 = st.columns(2)
        with col1:
            create_metric_card("Active Keys", f"{api_status['active_keys']}/{api_status['total_keys']}")
        with col2:
            create_metric_card("Current Key", f"#{api_status['current_key']}")
        
        # Status indicators
        for i, status in api_status['key_status'].items():
            status_type = "online" if status['active'] else "offline"
            create_status_indicator(status_type, f"API Key {i+1}")
        
        if st.button("🔄 Reset API Keys", key="reset_keys"):
            st.session_state.api_manager.reset_key_status()
            st.rerun()
        
        st.markdown("### 📈 Usage Quota")
        progress_value = quota_status['daily_used'] / max(quota_status['daily_limit'], 1)
        st.progress(progress_value)
        
        create_custom_card("Daily Usage", 
                          f"{quota_status['daily_used']}/{quota_status['daily_limit']} requests")
        create_custom_card("Hourly Usage", 
                          f"{quota_status['hourly_used']}/{quota_status['hourly_limit']} requests")
        create_custom_card("Session Requests", 
                          f"{quota_status['session_requests']} total")
    
    # Main Tabs with enhanced styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat", "📄 Document RAG", "🔧 Equipment Monitor", "📊 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        st.markdown("## 💬 Intelligent AI Assistant")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Query input with enhanced styling
        user_query = st.text_area("✨ Ask me anything:", height=120, 
                                  placeholder="Example: My TV is flickering, what should I check?")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🚀 Ask AI", type="primary", key="ask_ai"):
                if user_query:
                    # Check quota
                    can_make_request, quota_message = st.session_state.quota_manager.can_make_request()
                    
                    if not can_make_request:
                        create_alert_box(f"Quota exceeded: {quota_message}", "critical")
                    else:
                        with st.spinner("🤔 Analyzing your query..."):
                            # Process query
                            response = st.session_state.ai_engine.process_query(user_query)
                            
                            # Record successful request
                            if response['success']:
                                st.session_state.quota_manager.record_request()
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'query': user_query,
                                'response': response,
                                'timestamp': datetime.now()
                            })
                            
                            st.rerun()
        
        # Display chat history
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.container():
                st.markdown(f"""
                <div class="chat-message">
                    <div class="chat-query">🙋 {chat['query']}</div>
                    <div class="chat-response">🤖 {chat['response']['content']}</div>
                    <div class="chat-timestamp">
                        {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | 
                        Category: {chat['response']['classification']['category']} | 
                        Confidence: {chat['response']['confidence']:.2f}
                        {f" | API Key: #{chat['response'].get('api_key_used', 'N/A')}" if chat['response']['success'] else " | Fallback Mode"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show follow-up questions if available
                if chat['response'].get('follow_up_questions'):
                    with st.expander("💡 Suggested Follow-up Questions"):
                        for question in chat['response']['follow_up_questions']:
                            if st.button(question, key=f"followup_{i}_{hash(question)}"):
                                st.session_state.current_query = question
                                st.rerun()
    
    with tab2:
        st.markdown("## 📄 Document Knowledge Base (RAG)")
        
        # File upload
        uploaded_files = st.file_uploader(
            "📁 Upload Documents", 
            type=['pdf', 'txt', 'md', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload documents to build your knowledge base"
        )
        
        if uploaded_files:
            if st.button("📚 Process Documents", type="primary"):
                # Save uploaded files temporarily
                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(temp_path)
                
                # Process documents
                results = st.session_state.rag_system.add_documents(temp_files)
                
               # Continuation from the main() function - RAG tab completion

                # Clean up temp files
                for temp_path in temp_files:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Display results
                if results['success'] > 0:
                    create_alert_box(f"✅ Successfully processed {results['success']} documents with {results['total_chunks']} chunks", "success")
                if results['failed'] > 0:
                    create_alert_box(f"⚠️ Failed to process {results['failed']} documents", "warning")
        
        # Document search and query
        st.markdown("### 🔍 Query Your Knowledge Base")
        
        if len(st.session_state.rag_system.documents) > 0:
            st.success(f"📚 Knowledge Base: {len(st.session_state.rag_system.documents)} document chunks loaded")
            
            rag_query = st.text_area("🔍 Ask questions about your documents:", 
                                   placeholder="What is the main topic of the uploaded documents?")
            
            if st.button("🎯 Search & Generate Answer", key="rag_search"):
                if rag_query:
                    with st.spinner("🔍 Searching knowledge base..."):
                        response = st.session_state.rag_system.generate_answer(rag_query, st.session_state.ai_engine)
                        
                        # Display response
                        st.markdown("### 🤖 AI Response")
                        st.write(response['content'])
                        
                        # Display sources
                        if response.get('rag_sources'):
                            with st.expander(f"📖 Sources ({response['source_count']} found)"):
                                for i, source in enumerate(response['rag_sources'][:5], 1):
                                    st.markdown(f"""
                                    **Source {i}** (Similarity: {source['similarity_score']:.3f})
                                    
                                    {source['content'][:300]}...
                                    
                                    *Source: {source['source']} | Chunk: {source['chunk_id']}*
                                    """)
        else:
            create_alert_box("📋 No documents uploaded yet. Upload documents above to start building your knowledge base.", "info")
    
    with tab3:
        st.markdown("## 🔧 Equipment Monitoring & Diagnostics")
        
        # Equipment dashboard
        dashboard = st.session_state.maintenance_pipeline.get_equipment_dashboard()
        
        # Dashboard metrics
        st.markdown("### 📊 Fleet Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Total Equipment", dashboard['total_equipment'], "🏭")
        with col2:
            create_metric_card("Online", dashboard['online'], "🟢")
        with col3:
            create_metric_card("Offline", dashboard['offline'], "🔴")
        with col4:
            create_metric_card("Warnings", dashboard['warnings'], "⚠️")
        
        # Efficiency gauge
        st.markdown("### ⚡ Fleet Efficiency")
        efficiency = dashboard['avg_efficiency']
        
        # Create efficiency visualization
        efficiency_color = "🟢" if efficiency >= 90 else "🟡" if efficiency >= 75 else "🔴"
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); border-radius: 10px; color: white;">
            <h2>{efficiency_color} {efficiency:.1f}%</h2>
            <p>Average Fleet Efficiency</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Equipment list
        st.markdown("### 🏭 Equipment Status")
        
        for equipment in dashboard['equipment_list']:
            status_emoji = "🟢" if equipment['status'] == 'online' else "🟡" if equipment['status'] == 'standby' else "🔴"
            alert_badge = f"🚨 {equipment['alerts']} alerts" if equipment['alerts'] > 0 else "✅ No alerts"
            
            with st.container():
                st.markdown(f"""
                <div class="equipment-card">
                    <div class="equipment-header">
                        <h4>{status_emoji} {equipment['name']}</h4>
                        <span class="equipment-badge">{alert_badge}</span>
                    </div>
                    <div class="equipment-details">
                        <p><strong>ID:</strong> {equipment['id']} | <strong>Type:</strong> {equipment['type'].title()}</p>
                        <p><strong>Location:</strong> {equipment['location']}</p>
                        <p><strong>Efficiency:</strong> {equipment['efficiency']}% | <strong>Runtime:</strong> {equipment['runtime_hours']:,}h</p>
                        <p><strong>Next Maintenance:</strong> {equipment['next_maintenance']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Equipment actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"📋 Report", key=f"report_{equipment['id']}"):
                        report = st.session_state.maintenance_pipeline.generate_equipment_report(equipment['id'])
                        st.markdown(report)
                with col2:
                    if st.button(f"🔧 Diagnose", key=f"diagnose_{equipment['id']}"):
                        diagnostic_query = f"Equipment {equipment['name']} ({equipment['id']}) showing {equipment['alerts']} alerts with {equipment['efficiency']}% efficiency. What should I check?"
                        response = st.session_state.ai_engine.process_query(diagnostic_query)
                        st.write(f"**AI Diagnosis:** {response['content']}")
        
        # Maintenance due section
        st.markdown("### 📅 Maintenance Schedule")
        maintenance_due = dashboard['maintenance_due']
        
        if maintenance_due:
            for item in maintenance_due:
                urgency_color = {
                    'critical': '🔴',
                    'high': '🟠', 
                    'medium': '🟡'
                }.get(item['urgency'], '🟢')
                
                st.markdown(f"""
                <div class="maintenance-item">
                    <p>{urgency_color} <strong>{item['task']}</strong> - {item['equipment_id']}</p>
                    <p>Due: {item['next_due']} ({item['days_until_due']} days)</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No maintenance due in the next 7 days")
    
    with tab4:
        st.markdown("## 📊 Analytics & Insights")
        
        # Usage analytics
        st.markdown("### 📈 Usage Analytics")
        
        # Create sample analytics data
        dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
        usage_data = pd.DataFrame({
            'date': dates,
            'requests': np.random.poisson(15, len(dates)),
            'success_rate': np.random.uniform(0.85, 0.98, len(dates)),
            'avg_response_time': np.random.uniform(1.2, 3.5, len(dates))
        })
        
        # Usage over time chart
        st.markdown("#### 📊 Daily Requests")
        st.line_chart(usage_data.set_index('date')['requests'])
        
        # Success rate and response time
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ✅ Success Rate")
            st.line_chart(usage_data.set_index('date')['success_rate'])
        with col2:
            st.markdown("#### ⏱️ Response Time (seconds)")
            st.line_chart(usage_data.set_index('date')['avg_response_time'])
        
        # Query classification breakdown
        st.markdown("### 🎯 Query Classification Breakdown")
        
        classification_data = {
            'equipment_diagnostic': 35,
            'technical_support': 28,
            'general_inquiry': 20,
            'document_search': 12,
            'maintenance': 5
        }
        
        # Create pie chart data
        chart_data = pd.DataFrame(
            list(classification_data.items()),
            columns=['Category', 'Count']
        )
        
        # Display as horizontal bar chart
        st.bar_chart(chart_data.set_index('Category'))
        
        # Performance metrics
        st.markdown("### ⚡ Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            create_metric_card("Avg Response Time", "2.3s", "⏱️")
        with col2:
            create_metric_card("Success Rate", "94.2%", "✅")
        with col3:
            create_metric_card("Documents Processed", "127", "📄")
        with col4:
            create_metric_card("Equipment Monitored", f"{dashboard['total_equipment']}", "🏭")
    
    with tab5:
        st.markdown("## ⚙️ System Configuration")
        
        # API Configuration
        st.markdown("### 🔑 API Configuration")
        
        # Load current config
        current_config = load_config()
        
        with st.expander("🛠️ Model Settings"):
            # Model selection
            selected_model = st.selectbox(
                "Primary Model",
                options=['gemini-pro', 'gemini-pro-vision', 'text-bison'],
                index=['gemini-pro', 'gemini-pro-vision', 'text-bison'].index(current_config.model_name)
            )
            
            # Temperature setting
            temperature = st.slider(
                "Temperature (Creativity)",
                min_value=0.0,
                max_value=1.0,
                value=current_config.temperature,
                step=0.1,
                help="Higher values make responses more creative but less focused"
            )
            
            # Max tokens
            max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=100,
                max_value=8192,
                value=current_config.max_output_tokens,
                step=100
            )
        
        with st.expander("📊 RAG Settings"):
            # RAG configuration
            chunk_size = st.number_input(
                "Document Chunk Size (words)",
                min_value=100,
                max_value=1000,
                value=current_config.chunk_size,
                step=50
            )
            
            top_k_retrieval = st.number_input(
                "Top-K Retrieval",
                min_value=1,
                max_value=10,
                value=current_config.top_k_retrieval,
                step=1,
                help="Number of most relevant chunks to retrieve"
            )
            
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=current_config.similarity_threshold,
                step=0.1,
                help="Minimum similarity score to include a document chunk"
            )
        
        with st.expander("⏰ Quota Settings"):
            # Quota configuration
            daily_limit = st.number_input(
                "Daily Request Limit",
                min_value=10,
                max_value=1000,
                value=current_config.daily_limit,
                step=10
            )
            
            requests_per_hour = st.number_input(
                "Hourly Request Limit",
                min_value=5,
                max_value=100,
                value=current_config.requests_per_hour,
                step=5
            )
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            # Update configuration
            new_config = {
                'model_name': selected_model,
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'chunk_size': chunk_size,
                'top_k_retrieval': top_k_retrieval,
                'similarity_threshold': similarity_threshold,
                'daily_limit': daily_limit,
                'requests_per_hour': requests_per_hour
            }
            
            # Save to session state (in real app, save to file/database)
            for key, value in new_config.items():
                setattr(st.session_state.config, key, value)
            
            st.success("✅ Configuration saved successfully!")
            st.rerun()
        
        # System information
        st.markdown("### ℹ️ System Information")
        
        system_info = {
            'Python Version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'Streamlit Version': st.__version__,
            'Total Documents': len(st.session_state.rag_system.documents),
            'Active API Keys': st.session_state.api_manager.get_api_status()['active_keys'],
            'Session Duration': f"{(datetime.now() - datetime.now()).seconds // 60} minutes",
            'Cache Size': "N/A"  # Would implement actual cache size calculation
        }
        
        for key, value in system_info.items():
            st.markdown(f"**{key}:** {value}")
        
        # Database/Cache management
        st.markdown("### 🗃️ Data Management")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
        
        with col2:
            if st.button("📚 Clear Document Index"):
                st.session_state.rag_system.documents = []
                st.session_state.rag_system.index = None
                st.success("Document index cleared!")
        
        with col3:
            if st.button("🔄 Reset All Data"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    if key not in ['api_manager', 'quota_manager']:  # Keep essential managers
                        del st.session_state[key]
                st.success("All data reset!")
                st.rerun()
    
    # Footer
    st.markdown("""
    <div style="margin-top: 50px; padding: 20px; text-align: center; color: #666; border-top: 1px solid #eee;">
        <p>🤖 Versatile AI Assistant v2.0 | Enhanced Multi-Modal AI Platform</p>
        <p>Features: Equipment Diagnostics • Document RAG • Real-time Monitoring • Multi-API Support</p>
    </div>
    """, unsafe_allow_html=True)

# Additional utility functions for the complete application

def load_custom_css():
    """Load custom CSS for enhanced UI styling"""
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        color: #333;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Status indicators */
    .status-online {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0;
    }
    
    .status-offline {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0;
    }
    
    /* Chat styling */
    .chat-message {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .chat-query {
        background: #e0f2fe;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .chat-response {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .chat-timestamp {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Equipment cards */
    .equipment-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    .equipment-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .equipment-badge {
        background: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .equipment-details p {
        margin: 0.25rem 0;
        color: #4b5563;
    }
    
    /* Maintenance items */
    .maintenance-item {
        background: #fef9e7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    /* Alert boxes */
    .alert-success {
        background: #d1fae5;
        border: 1px solid #10b981;
        color: #065f46;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-error {
        background: #fee2e2;
        border: 1px solid #ef4444;
        color: #991b1b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background: #dbeafe;
        border: 1px solid #3b82f6;
        color: #1e40af;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        border: none;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #10b981, #059669);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, icon: str = "📊"):
    """Create a custom metric card with styling"""
    st.markdown(f"""
    <div class="metric-card">
        <h3>{icon} {value}</h3>
        <p>{title}</p>
    </div>
    """, unsafe_allow_html=True)

def create_status_indicator(status: str, label: str):
    """Create a status indicator with styling"""
    st.markdown(f"""
    <div class="status-{status}">
        {label}: {'🟢 Online' if status == 'online' else '🔴 Offline'}
    </div>
    """, unsafe_allow_html=True)

def create_custom_card(title: str, content: str):
    """Create a custom info card"""
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #3b82f6;">
        <strong>{title}</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)

def create_alert_box(message: str, alert_type: str):
    """Create styled alert boxes"""
    st.markdown(f"""
    <div class="alert-{alert_type}">
        {message}
    </div>
    """, unsafe_allow_html=True)

# Configuration classes and additional utilities

@dataclass
class Config:
    """Enhanced configuration class"""
    model_name: str = "gemini-pro"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    chunk_size: int = 300
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.7
    daily_limit: int = 100
    requests_per_hour: int = 20
    api_retry_attempts: int = 3
    api_retry_delay: int = 1
    enable_caching: bool = True
    log_level: str = "INFO"

@dataclass
class QuotaConfig:
    """Quota configuration"""
    daily_limit: int = 100
    requests_per_hour: int = 20
    rate_limit_window: int = 3600  # 1 hour in seconds
    burst_limit: int = 5  # Allow burst of 5 requests

def load_config() -> Config:
    """Load configuration from session state or defaults"""
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    return st.session_state.config

# Initialize quota configuration
quota_config = QuotaConfig()
config = load_config()

# Application entry point
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Versatile AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Run the main application
    main()              
