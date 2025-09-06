# app.py - Complete Enhanced version with quota management and fallback systems
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
from functools import wraps

# --- Enhanced Configuration with Quota Management ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Smart Equipment Diagnostic AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 3
    similarity_threshold: float = 0.4

@dataclass
class QuotaConfig:
    daily_limit: int = 50
    requests_per_hour: int = 10
    retry_delay: int = 60
    use_fallback_on_limit: bool = True

config = RAGConfig()
quota_config = QuotaConfig()

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

# --- Quota Management System ---
class QuotaManager:
    def __init__(self):
        self.request_log = self._load_request_log()
        self.session_requests = 0
    
    def _load_request_log(self) -> Dict:
        """Load request log from session state or create new one"""
        if 'quota_log' not in st.session_state:
            st.session_state.quota_log = {
                'daily_count': 0,
                'hourly_count': 0,
                'last_reset_date': datetime.now().date(),
                'last_reset_hour': datetime.now().hour,
                'total_requests': 0
            }
        return st.session_state.quota_log
    
    def can_make_request(self) -> tuple[bool, str]:
        """Check if we can make a request within quota limits"""
        now = datetime.now()
        
        # Reset daily counter if new day
        if now.date() > self.request_log['last_reset_date']:
            self.request_log['daily_count'] = 0
            self.request_log['last_reset_date'] = now.date()
        
        # Reset hourly counter if new hour
        if now.hour != self.request_log['last_reset_hour']:
            self.request_log['hourly_count'] = 0
            self.request_log['last_reset_hour'] = now.hour
        
        # Check daily limit
        if self.request_log['daily_count'] >= quota_config.daily_limit:
            return False, f"Daily quota exceeded ({quota_config.daily_limit} requests). Resets at midnight."
        
        # Check hourly limit
        if self.request_log['hourly_count'] >= quota_config.requests_per_hour:
            return False, f"Hourly quota exceeded ({quota_config.requests_per_hour} requests). Try again in {60 - now.minute} minutes."
        
        return True, "OK"
    
    def record_request(self, success: bool = True):
        """Record a request attempt"""
        if success:
            self.request_log['daily_count'] += 1
            self.request_log['hourly_count'] += 1
            self.request_log['total_requests'] += 1
            self.session_requests += 1
            st.session_state.quota_log = self.request_log
    
    def get_quota_status(self) -> Dict:
        """Get current quota usage status"""
        return {
            'daily_used': self.request_log['daily_count'],
            'daily_limit': quota_config.daily_limit,
            'hourly_used': self.request_log['hourly_count'],
            'hourly_limit': quota_config.requests_per_hour,
            'session_requests': self.session_requests,
            'total_requests': self.request_log['total_requests']
        }

# --- Rate Limiting Decorator ---
def rate_limit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        quota_manager = st.session_state.get('quota_manager')
        if not quota_manager:
            quota_manager = QuotaManager()
            st.session_state.quota_manager = quota_manager
        
        can_request, message = quota_manager.can_make_request()
        if not can_request:
            if quota_config.use_fallback_on_limit:
                st.warning(f"⚠️ {message} Using fallback mode.")
                return None  # Signal to use fallback
            else:
                raise Exception(f"Quota limit reached: {message}")
        
        try:
            result = func(*args, **kwargs)
            quota_manager.record_request(success=True)
            return result
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                quota_manager.record_request(success=False)
                st.error(f"🚫 API Quota Exceeded: {e}")
                if quota_config.use_fallback_on_limit:
                    return None  # Signal to use fallback
            raise e
    return wrapper

# --- Enhanced Smart Diagnostic Engine with Fallback ---
class SmartDiagnosticEngine:
    def __init__(self, gemini_model):
        self.gemini_model = gemini_model
        self.diagnostic_history = []
        self.fallback_solutions = self._load_fallback_solutions()
    
    def _load_fallback_solutions(self) -> Dict:
        """Load pre-built diagnostic solutions for common issues"""
        return {
            'hvac': {
                'no cooling': """**HVAC No Cooling - Diagnostic Steps:**
1. **Safety First**: Turn off system at breaker
2. **Check Thermostat**: Verify settings, replace batteries
3. **Air Filter**: Check and replace if dirty/clogged
4. **Circuit Breaker**: Ensure HVAC breaker hasn't tripped
5. **Outdoor Unit**: Check for debris around condenser
6. **Refrigerant Lines**: Look for ice buildup or leaks
**Tools Needed**: Multimeter, replacement filter, basic tools
**Call Professional If**: Electrical issues, refrigerant problems, compressor failure""",
                
                'strange noises': """**HVAC Strange Noises - Diagnostic Steps:**
1. **Identify Noise Type**: Grinding, squealing, banging, or clicking
2. **Location**: Indoor unit, outdoor unit, or ductwork
3. **Check Loose Parts**: Tighten panels and components
4. **Belt Inspection**: Check for wear or misalignment
5. **Fan Blades**: Ensure they're clean and balanced
**Immediate Actions**: Turn off if grinding sounds occur
**Professional Help**: Motor bearing issues, compressor problems"""
            },
            'electrical': {
                'power outages': """**Electrical Power Outage - Diagnostic Steps:**
1. **Safety Warning**: Never touch exposed wires
2. **Check Circuit Breakers**: Look for tripped breakers
3. **GFCI Outlets**: Press test/reset buttons
4. **Main Electrical Panel**: Verify main breaker position
5. **Neighbor Check**: Confirm if it's localized issue
6. **Utility Company**: Contact if widespread outage
**Call Professional**: Any burning smells, sparks, or complex wiring issues""",
                
                'flickering lights': """**Flickering Lights - Diagnostic Steps:**
1. **Bulb Check**: Try different bulb in same fixture
2. **Switch Inspection**: Toggle switch firmly
3. **Circuit Load**: Turn off other devices on same circuit
4. **Connection Check**: Ensure bulb is properly seated
5. **Dimmer Issues**: Replace if using dimmer switches
**Professional Help**: Loose wiring, voltage fluctuations"""
            },
            'network': {
                'connection timeout': """**Network Connection Timeout - Diagnostic Steps:**
1. **Physical Check**: Verify all cables are connected
2. **Reboot Sequence**: Modem → Router → Device (wait 30 sec between)
3. **LED Status**: Check link lights on network equipment
4. **Cable Test**: Try different Ethernet cable
5. **WiFi Signal**: Check signal strength and interference
6. **DNS Settings**: Try 8.8.8.8 or 1.1.1.1
**Tools**: Network cable tester, WiFi analyzer app""",
                
                'slow speeds': """**Slow Network Speeds - Diagnostic Steps:**
1. **Speed Test**: Use speedtest.net from multiple devices
2. **Bandwidth Usage**: Check for heavy downloads/streaming
3. **Device Limit**: Test with single device connected
4. **WiFi Channel**: Switch to less congested channel
5. **Router Location**: Ensure central, elevated position
6. **Firmware Update**: Update router firmware
**Professional Help**: ISP issues, infrastructure problems"""
            }
        }
    
    @rate_limit
    def analyze_user_query_with_ai(self, query: str, equipment_context: Optional[Dict] = None) -> Dict:
        """AI-powered query analysis with rate limiting"""
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
}"""
        
        response = self.gemini_model.generate_content(f"{system_prompt}\n\nUser Query: {query}")
        response_text = response.text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return self._fallback_analysis(query)
    
    def analyze_user_query(self, query: str, equipment_context: Optional[Dict] = None) -> Dict:
        """Main analysis method with fallback"""
        try:
            # Try AI analysis first
            result = self.analyze_user_query_with_ai(query, equipment_context)
            if result:
                return result
        except Exception as e:
            st.warning(f"AI analysis unavailable: Using fallback analysis.")
        
        # Use fallback analysis
        return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict:
        """Rule-based fallback analysis"""
        query_lower = query.lower()
        equipment_type = "general"
        
        # Equipment type detection
        for eq_type, eq_data in EQUIPMENT_KNOWLEDGE.items():
            if any(keyword in query_lower for keyword in [eq_type, eq_data['name'].lower()]):
                equipment_type = eq_type
                break
        
        # Problem classification
        problem_patterns = {
            'no cooling': ['not cool', 'no cool', 'warm air', 'hot air'],
            'strange noises': ['noise', 'sound', 'grinding', 'squealing', 'banging'],
            'power outages': ['no power', 'power out', 'electricity out'],
            'flickering lights': ['flicker', 'dim', 'bright'],
            'connection timeout': ['timeout', 'connect', 'internet down'],
            'slow speeds': ['slow', 'speed', 'laggy', 'performance']
        }
        
        detected_problem = 'general issue'
        for problem, patterns in problem_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_problem = problem
                break
        
        # Urgency detection
        urgency = "medium"
        if any(word in query_lower for word in ['urgent', 'critical', 'emergency', 'fire', 'smoke', 'sparks']):
            urgency = "critical"
        elif any(word in query_lower for word in ['broken', 'failed', 'dead', 'not working', 'stopped']):
            urgency = "high"
        elif any(word in query_lower for word in ['slow', 'intermittent', 'sometimes']):
            urgency = "low"
        
        return {
            "equipment_type": equipment_type,
            "component": None,
            "problem_summary": detected_problem,
            "urgency": urgency,
            "keywords": query_lower.split()[:5],
            "confidence": 75,
            "analysis_method": "fallback"
        }
    
    @rate_limit
    def generate_diagnostic_solution_with_ai(self, query: str, analysis: Dict, rag_context: str = "") -> str:
        """AI-powered solution generation with rate limiting"""
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
        
        response = self.gemini_model.generate_content(diagnostic_prompt)
        return response.text
    
    def generate_diagnostic_solution(self, query: str, analysis: Dict, rag_context: str = "") -> str:
        """Main solution generation with fallback"""
        try:
            # Try AI solution first
            result = self.generate_diagnostic_solution_with_ai(query, analysis, rag_context)
            if result:
                return result
        except Exception as e:
            st.warning(f"AI solution generation unavailable. Using fallback solution.")
        
        # Use fallback solution
        return self._get_fallback_solution(analysis, query)
    
    def _get_fallback_solution(self, analysis: Dict, query: str) -> str:
        """Get pre-built solution based on analysis"""
        equipment_type = analysis.get('equipment_type', 'general')
        problem = analysis.get('problem_summary', 'general issue')
        
        # Try to find specific solution
        equipment_solutions = self.fallback_solutions.get(equipment_type, {})
        for known_problem, solution in equipment_solutions.items():
            if known_problem in problem.lower():
                return f"🤖 **Fallback Diagnostic Solution** (AI temporarily unavailable)\n\n{solution}"
        
        # Generic fallback solution
        equipment_info = EQUIPMENT_KNOWLEDGE.get(equipment_type, {})
        return f"""🤖 **Basic Diagnostic Solution** (AI temporarily unavailable)

**Equipment**: {equipment_info.get('name', 'Equipment')}
**Issue**: {problem}

**Basic Troubleshooting Steps:**
1. **Safety First**: Turn off power to the equipment if electrical
2. **Visual Inspection**: Look for obvious damage, loose connections, or wear
3. **Check Power**: Verify equipment is receiving power
4. **Reset/Restart**: Try turning equipment off and on
5. **Check Connections**: Ensure all cables/connections are secure
6. **User Manual**: Consult equipment manual for specific troubleshooting

**Common Components to Check:**
{', '.join(equipment_info.get('components', ['Basic components']))}

**When to Call Professional:**
- Any electrical work beyond basic checks
- Gas-related equipment issues  
- Safety concerns or unusual odors
- Equipment under warranty

**Prevention:**
- Regular maintenance and cleaning
- Replace filters and consumables as scheduled
- Keep equipment area clean and ventilated

*Note: This is a basic solution. AI-powered detailed diagnostics will be available when quota resets.*
"""
    
    def generate_followup_questions(self, query: str, analysis: Dict) -> List[str]:
        """Generate follow-up questions with fallback"""
        equipment_type = analysis.get('equipment_type', 'general')
        try:
            if hasattr(self, 'gemini_model') and self.gemini_model:
                # Try AI generation (without rate limiting for questions - they're less critical)
                followup_prompt = f"""Based on this equipment issue, generate 3-4 specific diagnostic questions to better understand the problem:
Equipment Type: {equipment_type}
Issue: {analysis.get('problem_summary', query)}
Generate practical questions that a technician would ask to diagnose the issue. Return only the questions, one per line, starting with '- '."""
                response = self.gemini_model.generate_content(followup_prompt)
                questions = [line.strip()[2:] for line in response.text.split('\n') if line.strip().startswith('- ')]
                if questions:
                    return questions[:4]
        except Exception:
            pass
        
        # Fallback questions
        return self._get_fallback_questions(equipment_type)
    
    def _get_fallback_questions(self, equipment_type: str) -> List[str]:
        """Fallback diagnostic questions"""
        common_questions = [
            "When did the problem first occur?",
            "Are there any error messages or warning lights?",
            "Has anything changed recently (maintenance, updates, etc.)?",
            "Is the problem constant or intermittent?"
        ]
        
        equipment_specific = {
            'hvac': [
                "What is the current temperature reading?", 
                "Are all vents blowing air?",
                "Do you hear any unusual noises?",
                "When was the air filter last changed?"
            ],
            'electrical': [
                "Are any circuit breakers tripped?", 
                "Do you smell anything burning?",
                "Are other electrical devices working on the same circuit?",
                "When did you first notice the electrical issue?"
            ],
            'network': [
                "Are the status LEDs on the device lit?", 
                "Can you ping the device?",
                "Are other devices on the network working?",
                "Have you tried restarting the router?"
            ],
            'server': [
                "What error messages appear during boot?", 
                "Are the fans running?",
                "Is the server accessible remotely?",
                "When did you last restart the server?"
            ]
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
        return f"CSV file '{ticket_file}' loaded with {len(df)} records. Basic analysis available in fallback mode."
    except Exception as e:
        return f"Error analyzing CSV data: {e}"

# --- Enhanced LLM Configuration with Error Handling ---
def initialize_gemini():
    """Initialize Gemini with proper error handling"""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            return model
        else:
            st.error("❌ GEMINI_API_KEY not found in secrets. Please add it to use AI features.")
            return None
    except Exception as e:
        st.error(f"❌ Error configuring Gemini API: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            st.error("🚫 API quota exceeded. Please check your billing or wait for quota reset.")
        return None

# --- Document Processing Functions (Basic versions for fallback) ---
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

# --- Enhanced Quota Display Component ---
def display_quota_status():
    """Display current quota usage in sidebar"""
    if 'quota_manager' in st.session_state:
        status = st.session_state.quota_manager.get_quota_status()
        
        with st.sidebar:
            st.markdown("### 📊 API Quota Status")
            
            # Daily quota
            daily_pct = (status['daily_used'] / status['daily_limit']) * 100
            st.progress(daily_pct / 100, text=f"Daily: {status['daily_used']}/{status['daily_limit']} ({daily_pct:.1f}%)")
            
            # Hourly quota
            hourly_pct = (status['hourly_used'] / status['hourly_limit']) * 100
            st.progress(hourly_pct / 100, text=f"Hourly: {status['hourly_used']}/{status['hourly_limit']} ({hourly_pct:.1f}%)")
            
            # Session info
            st.metric("Session Requests", status['session_requests'])
            st.metric("Total Requests", status['total_requests'])
            
            # Status indicators
            if daily_pct > 90:
                st.error("🚫 Daily quota nearly exhausted!")
            elif daily_pct > 70:
                st.warning("⚠️ High daily quota usage")
            else:
                st.success("✅ Quota healthy")
            
            # Configuration options
            with st.expander("⚙️ Quota Settings"):
                new_fallback = st.checkbox(
                    "Use fallback when quota exceeded", 
                    value=quota_config.use_fallback_on_limit,
                    help="Enable fallback solutions when API quota is reached"
                )
                if new_fallback != quota_config.use_fallback_on_limit:
                    quota_config.use_fallback_on_limit = new_fallback
                    st.success("Setting updated!")

# --- Session State Initialization ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": """<span style="font-size:1.3em">🔧 <b>Smart Equipment Diagnostic AI Ready!</b></span>
Welcome! I can help you with:
- <b>Any equipment issue</b> (HVAC, electrical, network, servers, industrial)
- <b>Intelligent problem analysis</b>
- <b>Step-by-step solutions</b>
- <b>Safety recommendations</b>
- <b>Preventive maintenance</b>
<b>Just describe your problem in plain English!</b>  
<i>e.g. "My air conditioner stopped working"</i>
""", 
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

# --- Response Handling Function ---
def handle_user_prompt(prompt: str):
    with st.spinner("🤖 Analyzing issue and generating solution..."):
        try:
            response_content = ""
            if st.session_state.diagnostic_engine:
                analysis = st.session_state.diagnostic_engine.analyze_user_query(prompt)
                
                if analysis.get('equipment_type') != 'general':
                     st.session_state.current_analysis = {
                         'equipment_name': EQUIPMENT_KNOWLEDGE.get(analysis['equipment_type'], {}).get('name', 'Unknown'),
                         'confidence': analysis.get('confidence', 75),
                         'components': EQUIPMENT_KNOWLEDGE.get(analysis['equipment_type'], {}).get('components', [])
                     }
                
                # Generate solution (with automatic fallback)
                response_content = st.session_state.diagnostic_engine.generate_diagnostic_solution(
                    prompt, analysis, ""
                )
                
                # Generate follow-up questions
                st.session_state.followup_questions = st.session_state.diagnostic_engine.generate_followup_questions(
                    prompt, analysis
                )
            else:
                response_content = "Diagnostic engine not available. Operating in basic mode."
                
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
    st.markdown(
        """
        <style>
        .stChatMessage.user {background-color: #e3f2fd;}
        .stChatMessage.assistant {background-color: #f1f8e9;}
        .stMetric {background: #f5f5f5; border-radius: 8px;}
        .stButton>button {width: 100%;}
        .quota-warning {background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0;}
        </style>
        """, unsafe_allow_html=True
    )

    initialize_session_state()

# Continuation of app.py - Complete Enhanced version with quota management and fallback systems

    # Initialize quota manager
    if 'quota_manager' not in st.session_state:
        st.session_state.quota_manager = QuotaManager()

    # Initialize components
    gemini_model = initialize_gemini()
    
    if 'diagnostic_engine' not in st.session_state or st.session_state.diagnostic_engine is None:
        st.session_state.diagnostic_engine = SmartDiagnosticEngine(gemini_model)
    
    if 'maintenance_pipeline' not in st.session_state:
        st.session_state.maintenance_pipeline = MaintenancePipeline()

    # Title and Header
    st.title("🔧 Smart Equipment Diagnostic AI")
    st.markdown("### Intelligent Equipment Troubleshooting & Maintenance Assistant")
    
    # Display quota status in sidebar
    display_quota_status()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 🛠️ System Status")
        
        # AI Status Indicator
        if gemini_model:
            st.success("✅ AI Engine Active")
        else:
            st.error("❌ AI Engine Unavailable")
            st.info("🔄 Using Fallback Mode")
        
        st.markdown("---")
        
        # Equipment Type Filter
        st.markdown("### 🏭 Equipment Categories")
        for eq_type, eq_data in EQUIPMENT_KNOWLEDGE.items():
            with st.expander(f"🔹 {eq_data['name']}"):
                st.write(f"**Components:** {', '.join(eq_data['components'][:3])}...")
                st.write(f"**Common Issues:** {len(eq_data['common_issues'])} types")
        
        st.markdown("---")
        
        # Current Analysis Display
        if st.session_state.current_analysis:
            st.markdown("### 🔍 Current Analysis")
            st.success(f"**Equipment:** {st.session_state.current_analysis['equipment_name']}")
            st.info(f"**Confidence:** {st.session_state.current_analysis['confidence']}%")
            
            if st.session_state.current_analysis['components']:
                st.markdown("**Key Components:**")
                for comp in st.session_state.current_analysis['components'][:5]:
                    st.write(f"• {comp.title()}")
        
        st.markdown("---")
        
        # Tools Section
        st.markdown("### 🧰 Additional Tools")
        
        if st.button("📊 Generate Maintenance Report"):
            with st.spinner("Generating maintenance report..."):
                st.markdown("#### 📋 Equipment Status Report")
                
                # Sample data for demonstration
                high_risk_count = sum(1 for eq in st.session_state.maintenance_pipeline.maintenance_data.values() if eq['risk_level'] == 'HIGH')
                medium_risk_count = sum(1 for eq in st.session_state.maintenance_pipeline.maintenance_data.values() if eq['risk_level'] == 'MEDIUM')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Risk", high_risk_count, delta=f"+{high_risk_count-5}" if high_risk_count > 5 else f"{high_risk_count-5}")
                with col2:
                    st.metric("Medium Risk", medium_risk_count, delta=f"+{medium_risk_count-8}" if medium_risk_count > 8 else f"{medium_risk_count-8}")
                
                st.success("✅ Report generated successfully!")
        
        if st.button("🚨 Simulate Real-time Alert"):
            alert = st.session_state.maintenance_pipeline.simulate_real_time_alert()
            severity_color = {
                "LOW": "🟢", "MEDIUM": "🟡", 
                "HIGH": "🟠", "CRITICAL": "🔴"
            }
            
            st.markdown(f"""
            **Alert Generated:**
            
            {severity_color.get(alert['severity'], '⚪')} **{alert['alert_type']}**
            
            **Device:** {alert['device_id']}
            
            **Location:** {alert['location']}
            
            **Time:** {alert['timestamp']}
            
            **Description:** {alert['description']}
            """)
        
        if st.button("📈 Analyze CSV Data"):
            result = analyze_csv_data("analyze maintenance data")
            st.info(result)

    # Main Chat Interface
    st.markdown("## 💬 Diagnostic Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
            # Display timestamp for recent messages
            if message.get("timestamp"):
                time_diff = datetime.now() - message["timestamp"]
                if time_diff.total_seconds() < 3600:  # Within last hour
                    st.caption(f"⏰ {message['timestamp'].strftime('%H:%M:%S')}")

    # Follow-up Questions Section
    if st.session_state.followup_questions:
        st.markdown("### ❓ Follow-up Questions")
        st.info("Click on any question to get more specific help:")
        
        cols = st.columns(min(len(st.session_state.followup_questions), 3))
        for idx, question in enumerate(st.session_state.followup_questions[:3]):
            with cols[idx]:
                if st.button(f"❓ {question}", key=f"followup_{idx}"):
                    # Add question as user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now()
                    })
                    
                    # Process the follow-up question
                    handle_user_prompt(question)
                    st.rerun()

    # Chat Input
    if prompt := st.chat_input("Describe your equipment issue (e.g., 'My HVAC system is making strange noises')"):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Handle the user prompt
        handle_user_prompt(prompt)
        st.rerun()

    # Quick Action Buttons
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("❄️ HVAC Issues"):
            quick_prompt = "My HVAC system is not cooling properly"
            st.session_state.messages.append({
                "role": "user",
                "content": quick_prompt,
                "timestamp": datetime.now()
            })
            handle_user_prompt(quick_prompt)
            st.rerun()
    
    with col2:
        if st.button("⚡ Electrical Problems"):
            quick_prompt = "I'm having electrical issues with flickering lights"
            st.session_state.messages.append({
                "role": "user",
                "content": quick_prompt,
                "timestamp": datetime.now()
            })
            handle_user_prompt(quick_prompt)
            st.rerun()
    
    with col3:
        if st.button("🌐 Network Issues"):
            quick_prompt = "My network connection keeps timing out"
            st.session_state.messages.append({
                "role": "user",
                "content": quick_prompt,
                "timestamp": datetime.now()
            })
            handle_user_prompt(quick_prompt)
            st.rerun()
    
    with col4:
        if st.button("🖥️ Server Problems"):
            quick_prompt = "My server is showing high CPU usage and overheating"
            st.session_state.messages.append({
                "role": "user",
                "content": quick_prompt,
                "timestamp": datetime.now()
            })
            handle_user_prompt(quick_prompt)
            st.rerun()

    # Additional Features Section
    st.markdown("---")
    st.markdown("## 🔧 Advanced Features")
    
    tab1, tab2, tab3 = st.tabs(["📊 Analytics", "📚 Knowledge Base", "⚙️ Settings"])
    
    with tab1:
        st.markdown("### 📈 System Analytics")
        
        # Create sample analytics data
        if st.session_state.maintenance_pipeline:
            equipment_data = st.session_state.maintenance_pipeline.maintenance_data
            
            # Risk level distribution
            risk_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for eq_data in equipment_data.values():
                risk_counts[eq_data['risk_level']] += 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 High Risk", risk_counts['HIGH'])
            with col2:
                st.metric("🟡 Medium Risk", risk_counts['MEDIUM'])
            with col3:
                st.metric("🟢 Low Risk", risk_counts['LOW'])
            
            # Equipment type distribution
            st.markdown("#### Equipment Distribution")
            type_counts = {}
            for eq_data in equipment_data.values():
                eq_type = eq_data['type']
                type_counts[eq_type] = type_counts.get(eq_type, 0) + 1
            
            for eq_type, count in type_counts.items():
                st.write(f"**{eq_type}:** {count} units")
    
    with tab2:
        st.markdown("### 📖 Equipment Knowledge Base")
        
        selected_equipment = st.selectbox(
            "Select equipment type for details:",
            options=list(EQUIPMENT_KNOWLEDGE.keys()),
            format_func=lambda x: EQUIPMENT_KNOWLEDGE[x]['name']
        )
        
        if selected_equipment:
            eq_info = EQUIPMENT_KNOWLEDGE[selected_equipment]
            
            st.markdown(f"#### {eq_info['name']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Components:**")
                for component in eq_info['components']:
                    st.write(f"• {component.title()}")
            
            with col2:
                st.markdown("**Common Issues:**")
                for issue in eq_info['common_issues']:
                    st.write(f"• {issue.title()}")
    
    with tab3:
        st.markdown("### ⚙️ System Configuration")
        
        # Configuration options
        st.markdown("#### Chat Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_chunk_size = st.slider(
                "Text Chunk Size", 
                min_value=200, 
                max_value=1000, 
                value=config.chunk_size,
                step=50,
                help="Size of text chunks for processing"
            )
            if new_chunk_size != config.chunk_size:
                config.chunk_size = new_chunk_size
                st.success("✅ Chunk size updated!")
        
        with col2:
            new_top_k = st.slider(
                "Top-K Retrieval", 
                min_value=1, 
                max_value=10, 
                value=config.top_k_retrieval,
                help="Number of relevant chunks to retrieve"
            )
            if new_top_k != config.top_k_retrieval:
                config.top_k_retrieval = new_top_k
                st.success("✅ Top-K updated!")
        
        st.markdown("#### System Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
                st.session_state.current_analysis = None
                st.session_state.followup_questions = []
                st.success("✅ Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset System State"):
                for key in ['current_analysis', 'followup_questions', 'chat_history']:
                    if key in st.session_state:
                        st.session_state[key] = None if key != 'followup_questions' else []
                st.success("✅ System state reset!")
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🔧 <b>Smart Equipment Diagnostic AI</b> | 
        Powered by Advanced AI & Machine Learning | 
        ⚡ Real-time Diagnostics & Preventive Maintenance
        <br>
        <small>Version 2.0 | Enhanced with Quota Management & Fallback Systems</small>
    </div>
    """, unsafe_allow_html=True)

# --- Application Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("🔄 Please refresh the page or contact support if the issue persists.")
        
        # Show debug info in development
        if st.checkbox("Show Debug Info"):
            st.exception(e)
