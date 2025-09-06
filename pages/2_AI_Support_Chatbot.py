# app.py - Enhanced Versatile AI Assistant with Multi-API Key Management
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
            'medical': ['medical', 'hospital', 'mri', 'x-ray', 'ultrasound', 'equipment', 'device']
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

# --- Smart Equipment Knowledge Base (Enhanced) ---
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
    },
    'automotive': {
        'name': 'Automotive Systems',
        'components': ['engine', 'transmission', 'brakes', 'electrical system', 'cooling system', 'fuel system'],
        'common_issues': ['engine won\'t start', 'overheating', 'brake problems', 'transmission issues', 'electrical faults']
    },
    'medical': {
        'name': 'Medical Equipment',
        'components': ['imaging systems', 'monitors', 'pumps', 'ventilators', 'sensors', 'power supplies'],
        'common_issues': ['calibration errors', 'connectivity issues', 'power failures', 'sensor malfunctions', 'software errors']
    }
}

# --- Enhanced Quota Management System ---
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
                'total_requests': 0,
                'api_key_usage': {i: 0 for i in range(5)}  # Track usage per API key
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
    
    def record_request(self, success: bool = True, api_key_index: int = 0):
        """Record a request attempt"""
        if success:
            self.request_log['daily_count'] += 1
            self.request_log['hourly_count'] += 1
            self.request_log['total_requests'] += 1
            self.request_log['api_key_usage'][api_key_index] = self.request_log['api_key_usage'].get(api_key_index, 0) + 1
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
            'total_requests': self.request_log['total_requests'],
            'api_key_usage': self.request_log.get('api_key_usage', {})
        }

# --- Rate Limiting Decorator (Enhanced) ---
def rate_limit_with_rotation(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        quota_manager = st.session_state.get('quota_manager')
        api_manager = st.session_state.get('api_manager')
        
        if not quota_manager:
            quota_manager = QuotaManager()
            st.session_state.quota_manager = quota_manager
        
        if not api_manager:
            api_manager = MultiAPIManager()
            st.session_state.api_manager = api_manager
        
        can_request, message = quota_manager.can_make_request()
        if not can_request:
            if quota_config.use_fallback_on_limit:
                st.warning(f"⚠️ {message} Using fallback mode.")
                return None  # Signal to use fallback
            else:
                raise Exception(f"Quota limit reached: {message}")
        
        try:
            result = func(*args, **kwargs)
            quota_manager.record_request(success=True, api_key_index=api_manager.current_key_index)
            return result
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                quota_manager.record_request(success=False, api_key_index=api_manager.current_key_index)
                st.error(f"🚫 API Error: {e}")
                if quota_config.use_fallback_on_limit:
                    return None  # Signal to use fallback
            raise e
    return wrapper

# --- Enhanced Versatile AI Engine ---
class VersatileAIEngine:
    def __init__(self, api_manager: MultiAPIManager):
        self.api_manager = api_manager
        self.query_classifier = QueryClassifier()
        self.fallback_solutions = self._load_fallback_solutions()
    
    def _load_fallback_solutions(self) -> Dict:
        """Load comprehensive fallback solutions for all categories"""
        return {
            'equipment_diagnostic': {
                'hvac': {
                    'no cooling': """**HVAC No Cooling - Diagnostic Steps:**
1. **Safety First**: Turn off system at breaker
2. **Check Thermostat**: Verify settings, replace batteries
3. **Air Filter**: Check and replace if dirty/clogged
4. **Circuit Breaker**: Ensure HVAC breaker hasn't tripped
5. **Outdoor Unit**: Check for debris around condenser
**Tools Needed**: Multimeter, replacement filter, basic tools
**Call Professional If**: Electrical issues, refrigerant problems""",
                    'general': """**HVAC System Troubleshooting:**
1. Check power supply and circuit breakers
2. Inspect thermostat settings and batteries
3. Examine air filters for clogs
4. Look for unusual noises or vibrations
5. Check for proper airflow from vents"""
                },
                'electrical': {
                    'power outages': """**Electrical Power Issues:**
1. **Safety Warning**: Never touch exposed wires
2. Check circuit breakers for trips
3. Test GFCI outlets (reset if needed)
4. Verify main electrical panel
5. Contact utility if widespread outage""",
                    'general': """**Electrical System Diagnosis:**
1. Safety first - turn off power at breaker
2. Check for visible damage or burning smell
3. Test outlets and switches
4. Inspect wiring connections
5. Call professional for complex issues"""
                }
            },
            'technical_support': {
                'software': """**Software Troubleshooting Guide:**
1. **Identify the Issue**: Note exact error messages
2. **Basic Steps**: Restart application/system
3. **Check Logs**: Look for error details in logs
4. **Update Software**: Ensure latest version installed
5. **Reinstall**: Uninstall and reinstall if needed
6. **Check Dependencies**: Verify required components
7. **Search Documentation**: Check official docs/forums""",
                'code_debug': """**Code Debugging Process:**
1. **Read Error Messages**: Understand what went wrong
2. **Check Syntax**: Look for typos, missing brackets
3. **Print/Log Values**: Add debug output
4. **Test Small Parts**: Isolate problematic sections
5. **Use Debugger**: Step through code line by line
6. **Check Documentation**: Verify correct usage
7. **Ask Community**: Stack Overflow, forums"""
            },
            'general_inquiry': {
                'how_to': """**General Problem-Solving Approach:**
1. **Define the Goal**: What exactly do you want to achieve?
2. **Research**: Look up reliable sources and tutorials
3. **Break It Down**: Divide into smaller, manageable steps
4. **Start Simple**: Begin with basic version, then improve
5. **Practice**: Hands-on experience is crucial
6. **Ask for Help**: Use forums, communities, experts
7. **Document**: Keep notes of what works""",
                'information': """**Information Research Strategy:**
1. **Use Reliable Sources**: Academic, official websites
2. **Cross-Reference**: Check multiple sources
3. **Check Dates**: Ensure information is current
4. **Consider Context**: Understand the full picture
5. **Fact-Check**: Verify claims with evidence
6. **Note Sources**: Keep track of where info came from"""
            }
        }
    
    @rate_limit_with_rotation
    def analyze_and_respond(self, query: str) -> str:
        """Main method to analyze query and generate appropriate response"""
        # Classify the query
        classification = self.query_classifier.classify_query(query)
        
        # Get working model
        model = self.api_manager.get_working_model()
        if not model:
            return self._generate_fallback_response(query, classification)
        
        # Generate response based on classification
        try:
            if classification['category'] == 'equipment_diagnostic':
                return self._generate_equipment_response(query, classification, model)
            elif classification['category'] == 'technical_support':
                return self._generate_technical_response(query, classification, model)
            else:
                return self._generate_general_response(query, classification, model)
        
        except Exception as e:
            st.warning(f"AI generation failed: {e}. Using fallback.")
            return self._generate_fallback_response(query, classification)
    
    def _generate_equipment_response(self, query: str, classification: Dict, model) -> str:
        """Generate equipment diagnostic response"""
        equipment_type = classification.get('subcategory', 'general')
        equipment_info = EQUIPMENT_KNOWLEDGE.get(equipment_type, {})
        
        prompt = f"""You are an expert equipment diagnostic technician. Analyze this equipment issue and provide a comprehensive solution:

**Query**: {query}
**Equipment Type**: {equipment_info.get('name', 'Equipment')}
**Components**: {equipment_info.get('components', [])}
**Common Issues**: {equipment_info.get('common_issues', [])}

Provide a structured response with:
1. **Safety Checks** (if applicable)
2. **Diagnostic Steps** (step-by-step)
3. **Possible Causes** (ranked by likelihood)  
4. **Solutions** (detailed instructions)
5. **Tools/Parts Needed**
6. **Prevention Tips**
7. **When to Call Professional**

Be specific, practical, and safety-focused."""

        response = model.generate_content(prompt)
        return f"🔧 **Equipment Diagnostic Solution**\n\n{response.text}"
    
    def _generate_technical_response(self, query: str, classification: Dict, model) -> str:
        """Generate technical support response"""
        prompt = f"""You are an experienced technical support specialist. Help solve this technical issue:

**Query**: {query}
**Category**: Technical Support - {classification.get('subcategory', 'General')}

Provide a structured response with:
1. **Problem Analysis** (what's likely happening)
2. **Quick Fixes** (simple solutions to try first)
3. **Detailed Troubleshooting** (step-by-step process)
4. **Root Cause Investigation** (deeper analysis)
5. **Prevention Strategies** (avoid future issues)
6. **Additional Resources** (documentation, tools, communities)

Be clear, practical, and include relevant code examples if applicable."""

        response = model.generate_content(prompt)
        return f"💻 **Technical Support Solution**\n\n{response.text}"
    
    def _generate_general_response(self, query: str, classification: Dict, model) -> str:
        """Generate general inquiry response"""
        prompt = f"""You are a knowledgeable and helpful AI assistant. Provide a comprehensive answer to this query:

**Query**: {query}
**Type**: {classification.get('subcategory', 'General Information')}

Structure your response appropriately based on the query type:
- For "how-to" questions: Provide step-by-step instructions
- For informational queries: Give comprehensive, factual information
- For explanations: Break down complex topics clearly
- For advice: Offer practical, actionable guidance

Make your response:
- Clear and well-organized
- Practical and actionable
- Include examples where helpful
- Provide additional resources if relevant"""

        response = model.generate_content(prompt)
        return f"🤖 **AI Assistant Response**\n\n{response.text}"
    
    def _generate_fallback_response(self, query: str, classification: Dict) -> str:
        """Generate fallback response when AI is unavailable"""
        category = classification['category']
        subcategory = classification.get('subcategory', 'general')
        
        # Try to find specific fallback solution
        category_solutions = self.fallback_solutions.get(category, {})
        
        if isinstance(category_solutions, dict):
            # Look for subcategory-specific solution
            if subcategory in category_solutions:
                solution = category_solutions[subcategory]
            else:
                # Look for general solution in this category
                solution = category_solutions.get('general', "")
            
            if solution:
                return f"🤖 **Fallback Solution** (AI temporarily unavailable)\n\n{solution}"
        
        # Generic fallback
        return f"""🤖 **Basic Assistant Response** (AI temporarily unavailable)

**Your Query**: {query}
**Category**: {category.replace('_', ' ').title()}

**General Guidance:**
1. **Break Down the Problem**: Identify specific aspects of your issue
2. **Research Reliable Sources**: Use official documentation, reputable websites
3. **Start with Basics**: Try simple solutions before complex ones
4. **Safety First**: If dealing with equipment, ensure power is off
5. **Document Steps**: Keep track of what you try
6. **Seek Expert Help**: For complex or safety-critical issues

**Next Steps:**
- Try rephrasing your question for more specific help
- Check relevant documentation or manuals
- Consider consulting with domain experts

*Note: Full AI-powered responses will be available when API services are restored.*"""
    
    def generate_followup_questions(self, query: str, classification: Dict) -> List[str]:
        """Generate context-appropriate follow-up questions"""
        category = classification['category']
        
        if category == 'equipment_diagnostic':
            return [
                "When did the problem first start?",
                "Are there any error messages or warning lights?",
                "Has any maintenance been performed recently?",
                "Is the issue constant or intermittent?"
            ]
        elif category == 'technical_support':
            return [
                "What error messages do you see exactly?",
                "What were you doing when the problem occurred?",
                "Have you made any recent changes to your system?",
                "Does the problem happen consistently?"
            ]
        else:
            return [
                "Can you provide more specific details?",
                "What's your experience level with this topic?",
                "Are there particular aspects you'd like to focus on?",
                "Do you need step-by-step instructions or general overview?"
            ]

# --- Enhanced Maintenance Pipeline ---
class MaintenancePipeline:
    def __init__(self):
        self.maintenance_data = self._load_maintenance_data()
        self.equipment_list = list(self.maintenance_data.keys())

    def _load_maintenance_data(self) -> Dict:
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'Network Switch', 
                          'Server', 'Industrial Motor', 'Medical Device', 'Automotive System']
        
        for i in range(35):
            eq_type = random.choice(equipment_types)
            fail_prob = random.uniform(0.1, 0.9)
            equipment_data[f"{eq_type.replace(' ','_')}_{i+1}"] = {
                'type': eq_type,
                'location': f"Building {random.choice(['A', 'B', 'C'])} - Floor {random.randint(1,5)} - {random.choice(['Room', 'Rack', 'Bay'])} {random.randint(100, 599)}",
                'failure_probability': fail_prob,
                'risk_level': 'HIGH' if fail_prob > 0.7 else 'MEDIUM' if fail_prob > 0.4 else 'LOW',
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 8000),
                'last_issue': random.choice(['Overheating', 'Power failure', 'Network timeout', 'Mechanical wear', 'Software error', 'Calibration drift'])
            }
        return equipment_data

    def simulate_real_time_alert(self) -> Dict:
        alert_types = [
            "Device Offline", "High Temperature", "High CPU Usage", "Memory Error", 
            "Disk Failure", "Network Timeout", "Power Supply Failure", "Cooling Fan Error", 
            "Configuration Error", "Sensor Malfunction", "Software Crash", "Security Alert"
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
            "description": f"{alert_type} detected on {device_info['type']} at {device_info['location']}",
            "recommended_action": self._get_recommended_action(alert_type)
        }
    
    def _get_recommended_action(self, alert_type: str) -> str:
        actions = {
            "Device Offline": "Check network connectivity and power supply",
            "High Temperature": "Verify cooling systems and clean air filters", 
            "High CPU Usage": "Check for resource-intensive processes",
            "Memory Error": "Run memory diagnostics and check RAM modules",
            "Disk Failure": "Backup data immediately and replace disk",
            "Network Timeout": "Check network cables and switch ports",
            "Power Supply Failure": "Check power connections and replace PSU",
            "Cooling Fan Error": "Inspect and replace faulty fans",
            "Configuration Error": "Review and correct system settings",
            "Sensor Malfunction": "Calibrate or replace faulty sensors",
            "Software Crash": "Check logs and restart services",
            "Security Alert": "Investigate and secure system"
        }
        return actions.get(alert_type, "Investigate and take appropriate action")

# --- Enhanced UI Components ---
def display_api_status():
    """Display API status and management in sidebar"""
    if 'api_manager' in st.session_state:
        api_status = st.session_state.api_manager.get_api_status()
        
        with st.sidebar:
            st.markdown("### 🔑 API Key Status")
            
            # Overall status
            status_color = "🟢" if api_status['active_keys'] > 2 else "🟡" if api_status['active_keys'] > 0 else "🔴"
            st.markdown(f"{status_color} **{api_status['active_keys']}/{api_status['total_keys']} Keys Active**")
            st.markdown(f"**Current Key**: #{api_status['current_key']}")
# Continuation of app.py - Enhanced Versatile AI Assistant

            # Individual key status (continuing from where the code was cut off)
            with st.expander("🔍 Detailed Key Status"):
                for i, (key_idx, status) in enumerate(api_status['key_status'].items()):
                    key_status_icon = "✅" if status['active'] else "❌"
                    current_marker = " 👈" if i == api_status['current_key'] - 1 else ""
                    
                    st.markdown(f"{key_status_icon} **Key {key_idx + 1}**{current_marker}")
                    if not status['active'] and status['last_error']:
                        st.markdown(f"   ⚠️ *{status['last_error'][:50]}...*")
                    st.markdown(f"   Errors: {status['error_count']}")
                    
            # Reset buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Current", help="Reset current API key status"):
                    st.session_state.api_manager.reset_key_status(api_status['current_key'] - 1)
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset All", help="Reset all API keys"):
                    st.session_state.api_manager.reset_key_status()
                    st.rerun()

def display_quota_status():
    """Display quota management in sidebar"""
    if 'quota_manager' in st.session_state:
        quota_status = st.session_state.quota_manager.get_quota_status()
        
        with st.sidebar:
            st.markdown("### 📊 Usage Quota")
            
            # Daily quota
            daily_pct = (quota_status['daily_used'] / quota_status['daily_limit']) * 100
            daily_color = "🔴" if daily_pct >= 90 else "🟡" if daily_pct >= 70 else "🟢"
            st.markdown(f"{daily_color} **Daily**: {quota_status['daily_used']}/{quota_status['daily_limit']}")
            st.progress(daily_pct / 100)
            
            # Hourly quota
            hourly_pct = (quota_status['hourly_used'] / quota_status['hourly_limit']) * 100
            hourly_color = "🔴" if hourly_pct >= 90 else "🟡" if hourly_pct >= 70 else "🟢"
            st.markdown(f"{hourly_color} **Hourly**: {quota_status['hourly_used']}/{quota_status['hourly_limit']}")
            st.progress(hourly_pct / 100)
            
            # Session and total
            st.markdown(f"**Session**: {quota_status['session_requests']}")
            st.markdown(f"**Total**: {quota_status['total_requests']}")
            
            # API key usage distribution
            if quota_status.get('api_key_usage'):
                with st.expander("🔑 Key Usage Distribution"):
                    for key_idx, usage_count in quota_status['api_key_usage'].items():
                        st.markdown(f"Key {key_idx + 1}: {usage_count} requests")

def display_equipment_monitor():
    """Display equipment monitoring dashboard"""
    st.markdown("## 🏭 Equipment Health Monitor")
    
    maintenance_pipeline = MaintenancePipeline()
    
    # Real-time alert simulation
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🚨 Live Alerts")
    
    with col2:
        if st.button("🔄 Refresh Alerts"):
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", key="auto_refresh_alerts")
    
    # Generate and display alerts
    for i in range(3):  # Show 3 recent alerts
        alert = maintenance_pipeline.simulate_real_time_alert()
        severity_colors = {
            "LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"
        }
        
        with st.container():
            st.markdown(f"""
            **{severity_colors[alert['severity']]} {alert['alert_type']}** - {alert['severity']}
            - **Device**: {alert['device_id']} ({alert['device_type']})
            - **Location**: {alert['location']}
            - **Time**: {alert['timestamp']}
            - **Action**: {alert['recommended_action']}
            """)
            st.markdown("---")
    
    # Equipment overview
    st.markdown("### 📋 Equipment Overview")
    
    # Create equipment dataframe
    equipment_df = pd.DataFrame.from_dict(maintenance_pipeline.maintenance_data, orient='index')
    equipment_df.reset_index(inplace=True)
    equipment_df.rename(columns={'index': 'equipment_id'}, inplace=True)
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Risk Level", ['HIGH', 'MEDIUM', 'LOW'], default=['HIGH', 'MEDIUM', 'LOW'])
    with col2:
        type_filter = st.multiselect("Equipment Type", equipment_df['type'].unique())
    with col3:
        location_filter = st.selectbox("Location Filter", ['All'] + sorted(equipment_df['location'].unique()))
    
    # Apply filters
    filtered_df = equipment_df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
    if type_filter:
        filtered_df = filtered_df[filtered_df['type'].isin(type_filter)]
    if location_filter != 'All':
        filtered_df = filtered_df[filtered_df['location'] == location_filter]
    
    # Display filtered results
    st.dataframe(
        filtered_df[['equipment_id', 'type', 'location', 'risk_level', 'failure_probability', 
                    'next_maintenance', 'maintenance_cost', 'last_issue']],
        use_container_width=True
    )
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equipment", len(filtered_df))
    with col2:
        high_risk_count = len(filtered_df[filtered_df['risk_level'] == 'HIGH'])
        st.metric("High Risk", high_risk_count)
    with col3:
        avg_failure_prob = filtered_df['failure_probability'].mean()
        st.metric("Avg Failure Prob", f"{avg_failure_prob:.1%}")
    with col4:
        total_maintenance_cost = filtered_df['maintenance_cost'].sum()
        st.metric("Total Maint. Cost", f"${total_maintenance_cost:,}")

def display_diagnostic_helper():
    """Display diagnostic helper interface"""
    st.markdown("## 🔧 Smart Diagnostic Assistant")
    
    # Initialize AI engine
    if 'ai_engine' not in st.session_state:
        if 'api_manager' not in st.session_state:
            st.session_state.api_manager = MultiAPIManager()
        st.session_state.ai_engine = VersatileAIEngine(st.session_state.api_manager)
    
    ai_engine = st.session_state.ai_engine
    
    # Query input
    st.markdown("### 💬 Describe your issue or question")
    
    # Predefined quick options
    st.markdown("**Quick Examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌡️ HVAC not cooling properly"):
            st.session_state.diagnostic_query = "My HVAC system is not cooling properly. The unit runs but no cold air comes out."
    
    with col2:
        if st.button("💻 Server high CPU usage"):
            st.session_state.diagnostic_query = "Server showing consistently high CPU usage above 90%. System is slow to respond."
    
    with col3:
        if st.button("⚡ Electrical outlet not working"):
            st.session_state.diagnostic_query = "Electrical outlet stopped working suddenly. No power to devices plugged in."
    
    # Main query input
    query = st.text_area(
        "Enter your detailed question or issue description:",
        value=st.session_state.get('diagnostic_query', ''),
        height=100,
        placeholder="Describe your problem in detail. Include symptoms, when it started, any error messages, etc."
    )
    
    # Analysis options
    col1, col2 = st.columns([3, 1])
    with col1:
        include_followup = st.checkbox("Include follow-up questions", value=True)
    with col2:
        if st.button("🔍 Analyze & Get Solution", disabled=not query.strip()):
            with st.spinner("🤖 Analyzing your issue..."):
                # Classify query
                classification = ai_engine.query_classifier.classify_query(query)
                
                # Display classification
                st.markdown("### 📊 Issue Classification")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Category**: {classification['category'].replace('_', ' ').title()}")
                with col2:
                    st.markdown(f"**Type**: {classification['subcategory'].replace('_', ' ').title()}")
                with col3:
                    st.markdown(f"**Confidence**: {classification['confidence']:.1%}")
                
                if classification['keywords']:
                    st.markdown(f"**Keywords Detected**: {', '.join(classification['keywords'])}")
                
                st.markdown("---")
                
                # Generate response
                try:
                    response = ai_engine.analyze_and_respond(query)
                    
                    if response:
                        st.markdown("### 🎯 Diagnostic Solution")
                        st.markdown(response)
                        
                        # Generate follow-up questions if requested
                        if include_followup:
                            st.markdown("### ❓ Follow-up Questions")
                            followup_questions = ai_engine.generate_followup_questions(query, classification)
                            for i, question in enumerate(followup_questions, 1):
                                st.markdown(f"{i}. {question}")
                    else:
                        st.error("Failed to generate response. Please try again or check API status.")
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    # Show fallback
                    fallback_response = ai_engine._generate_fallback_response(query, classification)
                    st.markdown("### 🤖 Fallback Solution")
                    st.markdown(fallback_response)
    
    # Clear query button
    if st.button("🗑️ Clear Query"):
        if 'diagnostic_query' in st.session_state:
            del st.session_state.diagnostic_query
        st.rerun()

def display_configuration_panel():
    """Display system configuration panel"""
    st.markdown("## ⚙️ System Configuration")
    
    with st.expander("🔧 RAG Configuration", expanded=False):
        st.markdown("### Retrieval Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider("Chunk Size", 100, 1000, config.chunk_size, 50)
            top_k = st.slider("Top K Retrieval", 1, 10, config.top_k_retrieval)
        
        with col2:
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, config.similarity_threshold, 0.05)
        
        if st.button("💾 Update RAG Config"):
            config.chunk_size = chunk_size
            config.top_k_retrieval = top_k
            config.similarity_threshold = similarity_threshold
            st.success("✅ RAG configuration updated!")
    
    with st.expander("📊 Quota Management", expanded=False):
        st.markdown("### Quota Limits")
        
        col1, col2 = st.columns(2)
        with col1:
            daily_limit = st.number_input("Daily Request Limit", 1, 1000, quota_config.daily_limit)
            hourly_limit = st.number_input("Hourly Request Limit", 1, 100, quota_config.requests_per_hour)
        
        with col2:
            retry_delay = st.number_input("Retry Delay (seconds)", 30, 300, quota_config.retry_delay)
            use_fallback = st.checkbox("Use Fallback on Limit", quota_config.use_fallback_on_limit)
            api_rotation = st.checkbox("Enable API Key Rotation", quota_config.api_key_rotation)
        
        if st.button("💾 Update Quota Config"):
            quota_config.daily_limit = daily_limit
            quota_config.requests_per_hour = hourly_limit
            quota_config.retry_delay = retry_delay
            quota_config.use_fallback_on_limit = use_fallback
            quota_config.api_key_rotation = api_rotation
            st.success("✅ Quota configuration updated!")
    
    with st.expander("🔑 API Key Management", expanded=False):
        st.markdown("### Current API Key Status")
        
        if 'api_manager' in st.session_state:
            api_status = st.session_state.api_manager.get_api_status()
            
            # Display detailed key information
            for key_idx, status in api_status['key_status'].items():
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Key {key_idx + 1}**")
                    
                    with col2:
                        status_text = "🟢 Active" if status['active'] else "🔴 Disabled"
                        st.markdown(status_text)
                    
                    with col3:
                        st.markdown(f"Errors: {status['error_count']}")
                    
                    with col4:
                        if st.button(f"Reset {key_idx + 1}", key=f"reset_key_{key_idx}"):
                            st.session_state.api_manager.reset_key_status(key_idx)
                            st.rerun()
                    
                    if status['last_error']:
                        st.markdown(f"   ⚠️ Last Error: {status['last_error']}")
                    
                    st.markdown("---")
        else:
            st.warning("API Manager not initialized.")

def main():
    """Main application function"""
    # Initialize session state
    if 'api_manager' not in st.session_state:
        st.session_state.api_manager = MultiAPIManager()
    
    if 'quota_manager' not in st.session_state:
        st.session_state.quota_manager = QuotaManager()
    
    # Main header
    st.title("🤖 Enhanced Versatile AI Assistant")
    st.markdown("*Advanced multi-API, equipment diagnostics, and intelligent support system*")
    
    # Sidebar with status displays
    display_api_status()
    display_quota_status()
    
    # Main navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Diagnostic Assistant", 
        "🏭 Equipment Monitor", 
        "⚙️ Configuration", 
        "📊 System Status"
    ])
    
    with tab1:
        display_diagnostic_helper()
    
    with tab2:
        display_equipment_monitor()
    
    with tab3:
        display_configuration_panel()
    
    with tab4:
        st.markdown("## 📊 System Status Overview")
        
        # API Status Summary
        if 'api_manager' in st.session_state:
            api_status = st.session_state.api_manager.get_api_status()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total API Keys", api_status['total_keys'])
            with col2:
                st.metric("Active Keys", api_status['active_keys'])
            with col3:
                st.metric("Current Key", api_status['current_key'])
        
        # Quota Status Summary
        if 'quota_manager' in st.session_state:
            quota_status = st.session_state.quota_manager.get_quota_status()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Daily Used", f"{quota_status['daily_used']}/{quota_status['daily_limit']}")
            with col2:
                st.metric("Hourly Used", f"{quota_status['hourly_used']}/{quota_status['hourly_limit']}")
            with col3:
                st.metric("Session Requests", quota_status['session_requests'])
            with col4:
                st.metric("Total Requests", quota_status['total_requests'])
        
        # System Health Indicators
        st.markdown("### 🏥 System Health")
        
        health_metrics = {
            "API Connectivity": "🟢 Healthy" if 'api_manager' in st.session_state else "🔴 Error",
            "Quota Management": "🟢 Active" if 'quota_manager' in st.session_state else "🔴 Inactive",
            "Diagnostic Engine": "🟢 Ready" if 'ai_engine' in st.session_state else "🟡 Initializing",
            "Equipment Monitor": "🟢 Online"
        }
        
        for metric, status in health_metrics.items():
            st.markdown(f"**{metric}**: {status}")
    
    # Footer
    st.markdown("---")
    st.markdown("*🚀 Enhanced Versatile AI Assistant - Multi-API Management System*")
    st.markdown("*Built with Streamlit, Google Gemini API, and advanced diagnostics*")

if __name__ == "__main__":
    main()
