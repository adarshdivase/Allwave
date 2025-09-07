# app.py - Enhanced Versatile AI Assistant with Improved CSS Styling

import streamlit as st
import pandas as pd
import os
import sys
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Generator, Optional, Tuple
import fitz  # PyMuPDF
from datetime import datetime
from dataclasses import dataclass
import google.generativeai as genai
from PIL import Image
import io
import time
import json
from functools import wraps
import traceback

# --- Enhanced Configuration ---
warnings.filterwarnings("ignore")

# --- Enhanced CSS Styling ---
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

    /* --- ADDED RULE FOR DATA USAGE TEXT --- */
    .card-content {
        color: #000000;
        font-weight: 500;
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
    
    .alert-info {
        background: #eff6ff;
        border-color: #93c5fd;
        color: #1e40af;
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
        border-radius: var(--radius-sm) 0 0 var(--radius-sm);
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

    /* --- ADDED RULE FOR FOOTER TEXT --- */
    .custom-footer p {
        color: #000000;
        margin: 0.25rem 0;
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

# --- Custom UI Component Functions ---
def create_custom_card(title, content, card_type="default"):
    type_classes = {
        "success": "status-success",
        "warning": "status-warning",
        "error": "status-error",
        "default": ""
    }
    st.markdown(f"""
    <div class="custom-card {type_classes.get(card_type, '')} fade-in">
        <div class="card-header">{title}</div>
        <div class="card-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(label, value, delta=None):
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
    type_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "ℹ️"}
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
    status_colors = {"online": "#22c55e", "offline": "#ef4444", "warning": "#f59e0b", "maintenance": "#8b5cf6"}
    color = status_colors.get(status.lower(), "#64748b")
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0;">
        <div style="width: 10px; height: 10px; border-radius: 50%; background: {color};" class="pulse"></div>
        <span style="font-weight: 500; color: #FFFFFF;">{label}</span>
    </div>
    """, unsafe_allow_html=True)

# --- Configuration Dataclasses ---
@dataclass
class Config:
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7
    max_output_tokens: int = 2048
    chunk_size: int = 500
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.4
    daily_limit: int = 100
    requests_per_hour: int = 20
    api_key_rotation: bool = True
    use_fallback_on_limit: bool = True

def load_config() -> Config:
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    return st.session_state.config

config = load_config()

# --- Multi-API Key Management System ---
class MultiAPIManager:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.key_status = {i: {'active': True, 'error_count': 0, 'last_error': None} for i in range(len(self.api_keys))}
        self.current_model = None
        self._initialize_current_model()
    
    def _load_api_keys(self) -> List[str]:
        keys = []
        # Check for GEMINI_API_KEY_1 through GEMINI_API_KEY_5
        for i in range(1, 6):
            key_name = f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY"
            if key_name in st.secrets:
                keys.append(st.secrets[key_name])
        if not keys:
            st.error("❌ No API keys found. Please add GEMINI_API_KEY to secrets.")
        return keys
    
    def _initialize_current_model(self):
        if self.api_keys and self.current_key_index < len(self.api_keys):
            try:
                genai.configure(api_key=self.api_keys[self.current_key_index])
                self.current_model = genai.GenerativeModel(config.model_name)
                return True
            except Exception as e:
                st.error(f"Failed to initialize API key {self.current_key_index + 1}: {e}")
                return False
        return False
    
    def get_working_model(self):
        max_attempts = len(self.api_keys)
        for _ in range(max_attempts):
            if self.key_status[self.current_key_index]['active']:
                try:
                    if not self.current_model:
                        self._initialize_current_model()
                    # Test the current model with a simple request
                    _ = self.current_model.generate_content("Test")
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
        if (current_status['error_count'] >= 3 or 'quota' in error_str or '429' in error_str or 'rate limit' in error_str):
            current_status['active'] = False
            st.warning(f"⚠️ API Key {self.current_key_index + 1} disabled due to: {error}")
    
    def _rotate_to_next_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.current_model = None
        self._initialize_current_model()
    
    def get_api_status(self) -> Dict:
        active_keys = sum(1 for status in self.key_status.values() if status['active'])
        return {'total_keys': len(self.api_keys), 'active_keys': active_keys, 'current_key': self.current_key_index + 1, 'key_status': self.key_status}
    
    def reset_key_status(self):
        self.key_status = {i: {'active': True, 'error_count': 0, 'last_error': None} for i in range(len(self.api_keys))}
        st.success("✅ API key status reset!")

# --- Query Classification System ---
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
                return {'category': 'equipment_diagnostic', 'subcategory': equipment_type, 'confidence': 0.9, 'keywords': [kw for kw in keywords if kw in query_lower]}
        if any(keyword in query_lower for keyword in self.technical_keywords):
            return {'category': 'technical_support', 'subcategory': 'software', 'confidence': 0.8, 'keywords': [kw for kw in self.technical_keywords if kw in query_lower]}
        if any(keyword in query_lower for keyword in self.general_keywords):
            return {'category': 'general_inquiry', 'subcategory': 'information', 'confidence': 0.7, 'keywords': [kw for kw in self.general_keywords if kw in query_lower]}
        return {'category': 'general_inquiry', 'subcategory': 'unknown', 'confidence': 0.5, 'keywords': []}

# --- Core Application Logic Classes ---
class QuotaManager:
    def __init__(self):
        self.daily_limit = config.daily_limit
        self.hourly_limit = config.requests_per_hour
        self.reset_quota_if_needed()
        if 'session_requests' not in st.session_state:
            st.session_state.session_requests = 0
    
    def reset_quota_if_needed(self):
        today = datetime.now().strftime("%Y-%m-%d")
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        if 'quota_date' not in st.session_state or st.session_state.quota_date != today:
            st.session_state.quota_date = today
            st.session_state.daily_requests = 0
        if 'quota_hour' not in st.session_state or st.session_state.quota_hour != current_hour:
            st.session_state.quota_hour = current_hour
            st.session_state.hourly_requests = 0
    
    def can_make_request(self) -> Tuple[bool, str]:
        self.reset_quota_if_needed()
        daily_used = getattr(st.session_state, 'daily_requests', 0)
        hourly_used = getattr(st.session_state, 'hourly_requests', 0)
        if daily_used >= self.daily_limit:
            return False, f"Daily limit exceeded ({daily_used}/{self.daily_limit})"
        if hourly_used >= self.hourly_limit:
            return False, f"Hourly limit exceeded ({hourly_used}/{self.hourly_limit})"
        return True, "OK"
    
    def record_request(self):
        st.session_state.daily_requests = getattr(st.session_state, 'daily_requests', 0) + 1
        st.session_state.hourly_requests = getattr(st.session_state, 'hourly_requests', 0) + 1
        st.session_state.session_requests += 1
    
    def get_quota_status(self) -> Dict:
        self.reset_quota_if_needed()
        return {'daily_limit': self.daily_limit, 'daily_used': getattr(st.session_state, 'daily_requests', 0), 'hourly_limit': self.hourly_limit, 'hourly_used': getattr(st.session_state, 'hourly_requests', 0), 'session_requests': st.session_state.session_requests}

class VersatileAIEngine:
    def __init__(self, api_manager: MultiAPIManager):
        self.api_manager = api_manager
        self.classifier = QueryClassifier()
        self.conversation_memory = []
        self.max_memory = 10
    
    def process_query(self, query: str, context: str = "", image_data=None) -> Dict:
        classification = self.classifier.classify_query(query)
        model = self.api_manager.get_working_model()
        if not model:
            return self._get_fallback_response(query, classification)
        
        try:
            enhanced_prompt = self._build_enhanced_prompt(query, classification, context)
            generation_config = genai.types.GenerationConfig(temperature=config.temperature, max_output_tokens=config.max_output_tokens)
            
            if image_data:
                response = model.generate_content([enhanced_prompt, image_data], generation_config=generation_config)
            else:
                response = model.generate_content(enhanced_prompt, generation_config=generation_config)
            
            processed_response = self._process_response(response.text, classification)
            self._update_memory(query, processed_response['content'])
            
            return {'success': True, 'content': processed_response['content'], 'classification': classification, 'confidence': processed_response['confidence'], 'sources': processed_response.get('sources', []), 'follow_up_questions': processed_response.get('follow_up', []), 'api_key_used': self.api_manager.current_key_index + 1}
        
        except Exception as e:
            self.api_manager._handle_api_error(e)
            return self._get_fallback_response(query, classification, str(e))

    def _build_enhanced_prompt(self, query: str, classification: Dict, context: str) -> str:
        base_prompt = f"You are a versatile AI assistant. Your specialty is {classification['category']}.\nUser Query: {query}"
        if classification['category'] == 'equipment_diagnostic':
            base_prompt += "\nProvide step-by-step troubleshooting, safety warnings, common causes, and when to call a professional."
        if context:
            base_prompt += f"\n\nAdditional Context:\n{context}"
        if self.conversation_memory:
            memory_context = "\n".join([f"Q: {item['query']}\nA: {item['response'][:200]}..." for item in self.conversation_memory[-3:]])
            base_prompt += f"\n\nConversation History:\n{memory_context}"
        return base_prompt

    def _process_response(self, response_text: str, classification: Dict) -> Dict:
        processed = {'content': response_text, 'confidence': classification['confidence'], 'sources': [], 'follow_up': []}
        if classification['category'] == 'equipment_diagnostic':
            processed['follow_up'] = ["What specific symptoms are you seeing?", "When did this issue start?", "Have you tried any troubleshooting already?"]
        return processed

    def _update_memory(self, query: str, response: str):
        self.conversation_memory.append({'query': query, 'response': response, 'timestamp': datetime.now()})
        if len(self.conversation_memory) > self.max_memory:
            self.conversation_memory.pop(0)
    
    def _get_fallback_response(self, query: str, classification: Dict, error: str = "") -> Dict:
        fallback_msg = "I'm currently unable to access my full capabilities. For equipment issues, please check power connections and refer to your manual. For other issues, please try again later."
        return {'success': False, 'content': fallback_msg, 'classification': classification, 'confidence': 0.3, 'error': error, 'fallback': True}

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = config.chunk_size
        self.overlap = 50
    
    def process_document(self, file_path: str, file_name: str) -> List[Dict]:
        file_ext = os.path.splitext(file_name)[1].lower()
        try:
            if file_ext == '.pdf':
                return self._process_pdf(file_path, file_name)
            elif file_ext in ['.txt', '.md']:
                return self._process_text(file_path, file_name)
            else:
                st.error(f"Unsupported file format: {file_ext}")
                return []
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: str, file_name: str) -> List[Dict]:
        chunks = []
        with open(file_path, 'rb') as f:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                page_chunks = self._create_chunks(text, f"{file_name} (Page {page_num + 1})")
                chunks.extend(page_chunks)
        return chunks
    
    def _process_text(self, file_path: str, file_name: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self._create_chunks(text, source=file_name)
    
    def _create_chunks(self, text: str, source: any = None) -> List[Dict]:
        chunks = []
        words = re.split(r'\s+', text)
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({'content': chunk_text, 'source': source, 'chunk_id': len(chunks), 'word_count': len(chunk_words), 'timestamp': datetime.now()})
        return chunks

class EnhancedRAGSystem:
    def __init__(self):
        self.encoder = None
        self.index = None
        self.documents = []
        self.doc_processor = DocumentProcessor()
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        try:
            with st.spinner("Loading embedding model..."):
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"❌ Failed to load embedding model: {e}")
    
    def add_documents(self, file_paths: List[Tuple[str, str]]) -> Dict:
        results = {'success': 0, 'failed': 0, 'total_chunks': 0}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (file_path, file_name) in enumerate(file_paths):
            try:
                status_text.text(f"Processing: {file_name}")
                chunks = self.doc_processor.process_document(file_path, file_name)
                if chunks:
                    self.documents.extend(chunks)
                    results['success'] += 1
                    results['total_chunks'] += len(chunks)
                else:
                    results['failed'] += 1
            except Exception as e:
                results['failed'] += 1
                st.error(f"❌ Failed to process {file_name}: {e}")
            progress_bar.progress((i + 1) / len(file_paths))
        
        if self.documents and self.encoder:
            self._build_index()
        status_text.empty()
        progress_bar.empty()
        return results
    
    def _build_index(self):
        try:
            with st.spinner("Building search index..."):
                texts = [doc['content'] for doc in self.documents]
                embeddings = self.encoder.encode(texts, show_progress_bar=True)
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings.astype('float32'))
            st.success(f"✅ Search index built with {len(texts)} documents")
        except Exception as e:
            st.error(f"❌ Failed to build search index: {e}")
    
    def search(self, query: str) -> List[Dict]:
        if not self.encoder or not self.index or not self.documents:
            return []
        try:
            query_embedding = self.encoder.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), config.top_k_retrieval)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score < config.similarity_threshold: # L2 distance: lower is better
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
            return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return []

    def generate_answer(self, query: str, ai_engine: VersatileAIEngine) -> Dict:
        relevant_docs = self.search(query)
        if not relevant_docs:
            return ai_engine.process_query(query, "No relevant documents found in knowledge base.")
        context = "Relevant Information:\n\n" + "\n\n".join([f"{i+1}. {doc['content'][:500]}..." for i, doc in enumerate(relevant_docs)])
        context += f"\n\nBased on the above information from {len(relevant_docs)} sources, answer the question."
        response = ai_engine.process_query(query, context)
        response['rag_sources'] = relevant_docs
        response['source_count'] = len(relevant_docs)
        return response

class MaintenancePipeline:
    def __init__(self):
        self.equipment_db = self._load_equipment_data()
    
    def _load_equipment_data(self) -> List[Dict]:
        return [
            {'id': 'HVAC-001', 'name': 'Main HVAC Unit', 'type': 'hvac', 'location': 'Building A - Roof', 'status': 'online', 'last_maintenance': '2025-08-15', 'next_maintenance': '2025-11-15', 'alerts': 2, 'efficiency': 87, 'runtime_hours': 2340},
            {'id': 'GEN-001', 'name': 'Backup Generator', 'type': 'electrical', 'location': 'Basement', 'status': 'standby', 'last_maintenance': '2025-07-01', 'next_maintenance': '2025-10-01', 'alerts': 0, 'efficiency': 92, 'runtime_hours': 156},
            {'id': 'NET-001', 'name': 'Main Network Switch', 'type': 'network', 'location': 'Server Room', 'status': 'online', 'last_maintenance': '2025-06-30', 'next_maintenance': '2025-12-30', 'alerts': 1, 'efficiency': 95, 'runtime_hours': 8760}
        ]
    
    def get_equipment_dashboard(self) -> Dict:
        total = len(self.equipment_db)
        online = sum(1 for eq in self.equipment_db if eq['status'] == 'online')
        warnings = sum(1 for eq in self.equipment_db if eq['alerts'] > 0)
        return {'total_equipment': total, 'online': online, 'offline': total - online, 'warnings': warnings, 'avg_efficiency': sum(eq['efficiency'] for eq in self.equipment_db) / total, 'equipment_list': self.equipment_db}

# --- Main Streamlit Application ---
def main():
    load_custom_css()
    
    # Initialize session state
    if 'api_manager' not in st.session_state: st.session_state.api_manager = MultiAPIManager()
    if 'quota_manager' not in st.session_state: st.session_state.quota_manager = QuotaManager()
    if 'ai_engine' not in st.session_state: st.session_state.ai_engine = VersatileAIEngine(st.session_state.api_manager)
    if 'rag_system' not in st.session_state: st.session_state.rag_system = EnhancedRAGSystem()
    if 'maintenance_pipeline' not in st.session_state: st.session_state.maintenance_pipeline = MaintenancePipeline()
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    
    st.markdown("""<div class="main-header">
        <h1>🤖 Versatile AI Assistant</h1>
        <p>Advanced AI with Diagnostics, RAG, and Real-time Monitoring</p>
    </div>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("""<div class="sidebar-header">
            <h3 style="margin: 0; font-weight: 600;">🛠️ Control Panel</h3>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("### 📊 API Status")
        api_status = st.session_state.api_manager.get_api_status()
        c1, c2 = st.columns(2)
        with c1: create_metric_card("Active Keys", f"{api_status['active_keys']}/{api_status['total_keys']}")
        with c2: create_metric_card("Current Key", f"#{api_status['current_key']}")
        for i, status in api_status['key_status'].items():
            create_status_indicator("online" if status['active'] else "offline", f"API Key {i+1}")
        if st.button("🔄 Reset API Keys", key="reset_keys"): st.session_state.api_manager.reset_key_status(); st.rerun()

        st.markdown("### 📈 Usage Quota")
        quota_status = st.session_state.quota_manager.get_quota_status()
        st.progress(quota_status['daily_used'] / max(quota_status['daily_limit'], 1))
        create_custom_card("Daily Usage", f"{quota_status['daily_used']}/{quota_status['daily_limit']} requests")
        create_custom_card("Hourly Usage", f"{quota_status['hourly_used']}/{quota_status['hourly_limit']} requests")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 AI Chat", "📄 Document RAG", "🔧 Equipment Monitor", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.markdown("## 💬 Intelligent AI Assistant")
        user_query = st.text_area("✨ Ask me anything:", height=120, placeholder="Example: My TV is flickering, what should I check?")
        if st.button("🚀 Ask AI", type="primary", use_container_width=True):
            if user_query:
                can_request, msg = st.session_state.quota_manager.can_make_request()
                if not can_request:
                    create_alert_box(f"Quota exceeded: {msg}", "critical")
                else:
                    with st.spinner("🤔 Thinking..."):
                        response = st.session_state.ai_engine.process_query(user_query)
                        if response.get('success', False):
                            st.session_state.quota_manager.record_request()
                        st.session_state.chat_history.append({'query': user_query, 'response': response, 'timestamp': datetime.now()})
                        st.rerun()

        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"""<div class="chat-message">
                <div class="chat-query">🙋 {chat['query']}</div>
                <div class="chat-response">🤖 {chat['response']['content']}</div>
                <div class="chat-timestamp">{chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | Category: {chat['response']['classification']['category']}</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("## 📄 Document Knowledge Base (RAG)")
        uploaded_files = st.file_uploader("📁 Upload Documents", type=['pdf', 'txt', 'md'], accept_multiple_files=True)
        if uploaded_files:
            if st.button("📚 Process Documents", type="primary"):
                temp_files = []
                for file in uploaded_files:
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    temp_files.append((temp_path, file.name))
                results = st.session_state.rag_system.add_documents(temp_files)
                for path, name in temp_files: os.remove(path)
                if results['success'] > 0: create_alert_box(f"✅ Processed {results['success']} documents with {results['total_chunks']} chunks.", "low")
        
        if len(st.session_state.rag_system.documents) > 0:
            st.success(f"📚 Knowledge Base: {len(st.session_state.rag_system.documents)} document chunks loaded.")
            rag_query = st.text_area("🔍 Ask questions about your documents:")
            if st.button("🎯 Search & Generate Answer", key="rag_search"):
                if rag_query:
                    with st.spinner("🔍 Searching knowledge base..."):
                        response = st.session_state.rag_system.generate_answer(rag_query, st.session_state.ai_engine)
                        st.markdown("### 🤖 AI Response"); st.write(response['content'])
                        if response.get('rag_sources'):
                            with st.expander(f"📖 Sources ({response['source_count']} found)"):
                                for source in response['rag_sources']:
                                    st.info(f"**Source:** {source['source']} (Score: {source['similarity_score']:.3f})\n\n{source['content'][:300]}...")
        else:
            create_alert_box("No documents uploaded yet. Upload documents to build your knowledge base.", "info")
            
    with tab3:
        st.markdown("## 🔧 Equipment Monitoring")
        dashboard = st.session_state.maintenance_pipeline.get_equipment_dashboard()
        c1, c2, c3, c4 = st.columns(4)
        with c1: create_metric_card("Total Equipment", dashboard['total_equipment'])
        with c2: create_metric_card("Online", dashboard['online'])
        with c3: create_metric_card("Offline", dashboard['offline'])
        with c4: create_metric_card("Warnings", dashboard['warnings'])
        
        st.markdown("### 🏭 Equipment Status")
        for eq in dashboard['equipment_list']:
            with st.container(border=True):
                c1,c2 = st.columns([3,1])
                c1.subheader(f"{'🟢' if eq['status']=='online' else '🔴'} {eq['name']}")
                c1.text(f"ID: {eq['id']} | Location: {eq['location']}")
                c2.metric("Efficiency", f"{eq['efficiency']}%")
                if st.button(f"🔧 Diagnose", key=f"diagnose_{eq['id']}"):
                    query = f"Diagnose issue with {eq['name']} (efficiency: {eq['efficiency']}%, alerts: {eq['alerts']})"
                    with st.spinner("Running diagnostics..."):
                        st.write(st.session_state.ai_engine.process_query(query)['content'])

    with tab4:
        st.markdown("## 📊 Analytics & Insights")
        st.markdown("### 🎯 Query Classification Breakdown")
        if st.session_state.chat_history:
            class_counts = pd.Series([c['response']['classification']['category'] for c in st.session_state.chat_history]).value_counts()
            st.bar_chart(class_counts)
        else:
            st.info("No queries made in this session yet.")
            
    with tab5:
        st.markdown("## ⚙️ System Configuration")
        with st.expander("🛠️ RAG & Model Settings"):
            config.chunk_size = st.number_input("Chunk Size", 100, 1000, config.chunk_size, 50)
            config.top_k_retrieval = st.slider("Top-K Retrieval", 1, 10, config.top_k_retrieval)
            config.similarity_threshold = st.slider("Similarity Threshold (lower is better)", 0.0, 5.0, config.similarity_threshold, 0.1)
            config.temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, config.temperature, 0.1)

        with st.expander("⏰ Quota Settings"):
            config.daily_limit = st.number_input("Daily Request Limit", 10, 1000, config.daily_limit, 10)
            config.requests_per_hour = st.number_input("Hourly Request Limit", 5, 100, config.requests_per_hour, 5)

        if st.button("💾 Save Configuration"):
            st.session_state.config = config # Save updated config
            # Re-initialize components that depend on config
            st.session_state.quota_manager = QuotaManager() 
            st.session_state.rag_system.doc_processor.chunk_size = config.chunk_size
            st.success("✅ Configuration saved!")

        st.markdown("### 🗃️ Data Management")
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear Chat History"): st.session_state.chat_history = []; st.success("Chat history cleared!"); st.rerun()
        if c2.button("📚 Clear Document Index"): st.session_state.rag_system = EnhancedRAGSystem(); st.success("Document index cleared!"); st.rerun()

    st.markdown("""<div class="custom-footer">
        <p>🤖 Versatile AI Assistant v1.0</p>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Versatile AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main()
