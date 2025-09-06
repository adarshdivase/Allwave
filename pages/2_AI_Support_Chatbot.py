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
    },
    'television': {
        'name': 'Television/Display',
        'components': ['display panel', 'backlight', 'power supply', 'main board', 'T-con board', 'speakers', 'remote sensor', 'HDMI ports'],
        'common_issues': ['flickering screen', 'no picture', 'sound but no image', 'lines on screen', 'color issues', 'won\'t turn on', 'remote not working', 'connectivity problems']
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
            # Ensure api_key_usage dict exists
            if 'api_key_usage' not in self.request_log:
                self.request_log['api_key_usage'] = {i: 0 for i in range(5)}
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
            # Record successful request
            quota_manager.record_request(success=True, api_key_index=api_manager.current_key_index)
            return result
        except Exception as e:
            error_str = str(e).lower()
            # Record failed request but don't count against quota for failures
            
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                st.error(f"🚫 API Rate Limit: {e}")
                if quota_config.use_fallback_on_limit:
                    return None  # Signal to use fallback
            else:
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
                'television': {
                    'flickering screen': """**TV Flickering Screen - Diagnostic Steps:**
1. **Safety First**: Unplug TV for 30 seconds, then plug back in
2. **Check Connections**: Ensure all cables (HDMI, power) are secure
3. **Test Different Sources**: Try different HDMI ports or inputs
4. **Check Refresh Rate**: Ensure source device matches TV's supported rates
5. **Power Supply**: Look for capacitor issues (bulging caps on power board)
6. **Backlight Test**: Shine flashlight on screen - if you see faint image, backlight issue
**Tools Needed**: Flashlight, multimeter (for advanced troubleshooting)
**Call Professional If**: Internal board issues, backlight replacement needed""",
                    'general': """**TV/Display Troubleshooting:**
1. Check all cable connections (HDMI, power, etc.)
2. Test with different input sources
3. Verify TV and source device settings match
4. Check for software/firmware updates
5. Reset TV to factory settings if needed
6. Test with different cables
7. Check power supply and internal boards"""
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
        
# Continuing from the MaintenancePipeline class...

        return {
            "alert_type": alert_type,
            "device_id": device,
            "device_type": device_info['type'],
            "location": device_info['location'],
            "severity": random.choice(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "failure_probability": device_info['failure_probability'],
            "estimated_cost": random.randint(500, 5000),
            "recommended_action": self._get_recommended_action(alert_type),
            "urgency_hours": random.randint(1, 72)
        }

    def _get_recommended_action(self, alert_type: str) -> str:
        actions = {
            "Device Offline": "Check network connectivity and power supply",
            "High Temperature": "Inspect cooling systems and clean air filters",
            "High CPU Usage": "Analyze running processes and optimize workload",
            "Memory Error": "Run memory diagnostics and consider replacement",
            "Disk Failure": "Backup data immediately and replace disk",
            "Network Timeout": "Check network cables and switch configuration",
            "Power Supply Failure": "Replace power supply unit immediately",
            "Cooling Fan Error": "Replace or clean cooling fans",
            "Configuration Error": "Review and restore proper configuration",
            "Sensor Malfunction": "Calibrate or replace faulty sensor",
            "Software Crash": "Restart service and check error logs",
            "Security Alert": "Investigate potential security breach"
        }
        return actions.get(alert_type, "Contact technical support")

    def get_maintenance_dashboard(self) -> Dict:
        high_risk = sum(1 for eq in self.maintenance_data.values() if eq['risk_level'] == 'HIGH')
        medium_risk = sum(1 for eq in self.maintenance_data.values() if eq['risk_level'] == 'MEDIUM')
        low_risk = sum(1 for eq in self.maintenance_data.values() if eq['risk_level'] == 'LOW')
        
        return {
            'total_equipment': len(self.maintenance_data),
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'equipment_details': self.maintenance_data
        }

# --- Document Processing System ---
class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx', '.eml', '.mbox']
    
    def process_uploaded_files(self, uploaded_files) -> List[Dict]:
        """Process multiple uploaded files and extract text content"""
        processed_docs = []
        
        for uploaded_file in uploaded_files:
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                if file_extension == '.pdf':
                    content = self._extract_from_pdf(uploaded_file)
                elif file_extension == '.txt':
                    content = self._extract_from_txt(uploaded_file)
                elif file_extension == '.eml':
                    content = self._extract_from_eml(uploaded_file)
                elif file_extension == '.mbox':
                    content = self._extract_from_mbox(uploaded_file)
                else:
                    content = f"Unsupported file format: {file_extension}"
                
                processed_docs.append({
                    'filename': uploaded_file.name,
                    'content': content,
                    'size': uploaded_file.size,
                    'type': file_extension,
                    'processed_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                processed_docs.append({
                    'filename': uploaded_file.name,
                    'content': f"Error processing file: {str(e)}",
                    'size': uploaded_file.size if hasattr(uploaded_file, 'size') else 0,
                    'type': 'error',
                    'processed_at': datetime.now().isoformat()
                })
        
        return processed_docs
    
    def _extract_from_pdf(self, uploaded_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text() + "\n"
            pdf_document.close()
            return text
        except Exception as e:
            return f"Error extracting PDF content: {str(e)}"
    
    def _extract_from_txt(self, uploaded_file) -> str:
        """Extract text from TXT file"""
        try:
            return str(uploaded_file.read(), "utf-8")
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    
    def _extract_from_eml(self, uploaded_file) -> str:
        """Extract content from EML email file"""
        try:
            email_bytes = uploaded_file.read()
            msg = BytesParser(policy=policy.default).parsebytes(email_bytes)
            
            content = f"Subject: {msg['subject']}\n"
            content += f"From: {msg['from']}\n"
            content += f"To: {msg['to']}\n"
            content += f"Date: {msg['date']}\n\n"
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                content += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return content
        except Exception as e:
            return f"Error processing EML file: {str(e)}"
    
    def _extract_from_mbox(self, uploaded_file) -> str:
        """Extract content from MBOX file"""
        try:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            mbox = mailbox.mbox(temp_path)
            content = ""
            
            for i, message in enumerate(mbox):
                if i >= 10:  # Limit to first 10 emails
                    break
                content += f"\n--- Email {i+1} ---\n"
                content += f"Subject: {message['subject']}\n"
                content += f"From: {message['from']}\n"
                content += f"Date: {message['date']}\n\n"
                
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                payload = part.get_payload(decode=True)
                                if payload:
                                    content += payload.decode('utf-8', errors='ignore')
                            except:
                                content += "[Unable to decode content]"
                else:
                    try:
                        payload = message.get_payload(decode=True)
                        if payload:
                            content += payload.decode('utf-8', errors='ignore')
                    except:
                        content += "[Unable to decode content]"
            
            # Clean up temp file
            os.remove(temp_path)
            return content
        except Exception as e:
            return f"Error processing MBOX file: {str(e)}"

# --- RAG System with FAISS ---
class EnhancedRAGSystem:
    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.document_chunks = []
        self.is_initialized = False
    
    def initialize_embeddings(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.is_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {e}")
            return False
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def build_index(self, documents: List[Dict]):
        """Build FAISS index from processed documents"""
        if not self.is_initialized:
            if not self.initialize_embeddings():
                return False
        
        try:
            self.documents = documents
            self.document_chunks = []
            
            # Extract and chunk text from all documents
            for doc in documents:
                chunks = self.chunk_text(doc['content'], config.chunk_size)
                for i, chunk in enumerate(chunks):
                    self.document_chunks.append({
                        'text': chunk,
                        'doc_filename': doc['filename'],
                        'chunk_id': i,
                        'doc_type': doc['type']
                    })
            
            if not self.document_chunks:
                st.warning("No text chunks extracted from documents")
                return False
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in self.document_chunks]
            embeddings = self.model.encode(texts)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            return True
            
        except Exception as e:
            st.error(f"Failed to build RAG index: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant document chunks"""
        if not self.is_initialized or self.index is None:
            return []
        
        try:
            top_k = top_k or config.top_k_retrieval
            
            # Generate query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= config.similarity_threshold:  # Filter by similarity threshold
                    chunk_info = self.document_chunks[idx].copy()
                    chunk_info['similarity_score'] = float(score)
                    results.append(chunk_info)
            
            return results
            
        except Exception as e:
            st.error(f"RAG search failed: {e}")
            return []
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query"""
        search_results = self.search(query)
        
        if not search_results:
            return ""
        
        context_parts = []
        for result in search_results:
            context_parts.append(f"From {result['doc_filename']}:\n{result['text']}")
        
        return "\n\n".join(context_parts)

# --- Main Streamlit Application ---
def main():
    # Initialize session state
    if 'api_manager' not in st.session_state:
        st.session_state.api_manager = MultiAPIManager()
    
    if 'quota_manager' not in st.session_state:
        st.session_state.quota_manager = QuotaManager()
    
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = VersatileAIEngine(st.session_state.api_manager)
    
    if 'maintenance_pipeline' not in st.session_state:
        st.session_state.maintenance_pipeline = MaintenancePipeline()
    
    if 'document_processor' not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedRAGSystem()
    
    # App Header
    st.title("🤖 Versatile AI Assistant")
    st.markdown("*Advanced Multi-Modal AI with Equipment Diagnostics, RAG, and Real-time Monitoring*")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Control Panel")
        
        # API Status
        api_status = st.session_state.api_manager.get_api_status()
        quota_status = st.session_state.quota_manager.get_quota_status()
        
        st.subheader("📊 API Status")
        st.metric("Active Keys", f"{api_status['active_keys']}/{api_status['total_keys']}")
        st.metric("Current Key", f"#{api_status['current_key']}")
        
        if st.button("🔄 Reset API Keys"):
            st.session_state.api_manager.reset_key_status()
        
        st.subheader("📈 Usage Quota")
        st.progress(quota_status['daily_used'] / quota_status['daily_limit'])
        st.text(f"Daily: {quota_status['daily_used']}/{quota_status['daily_limit']}")
        st.text(f"Hourly: {quota_status['hourly_used']}/{quota_status['hourly_limit']}")
        st.text(f"Session: {quota_status['session_requests']}")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        config.chunk_size = st.slider("RAG Chunk Size", 200, 1000, config.chunk_size)
        config.top_k_retrieval = st.slider("Top-K Retrieval", 1, 10, config.top_k_retrieval)
        config.similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, config.similarity_threshold)
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 AI Chat", "📄 Document RAG", "🔧 Equipment Monitor", "📊 Analytics", "⚙️ Settings"
    ])
    
    with tab1:
        st.header("💬 Intelligent AI Assistant")
        
        # Chat interface
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Query input
        user_query = st.text_area("Ask me anything:", height=100, 
                                 placeholder="Example: My TV is flickering, what should I check?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Ask AI", type="primary"):
                if user_query:
                    with st.spinner("🤔 Analyzing your query..."):
                        # Get response from AI engine
                        response = st.session_state.ai_engine.analyze_and_respond(user_query)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'query': user_query,
                            'response': response,
                            'timestamp': datetime.now()
                        })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['query'][:50]}...", expanded=(i==0)):
                    st.markdown(f"**Question:** {chat['query']}")
                    st.markdown(f"**Response:** {chat['response']}")
                    st.text(f"Asked: {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    with tab2:
        st.header("📄 Document Knowledge Base (RAG)")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents for knowledge base:",
            type=['pdf', 'txt', 'eml', 'mbox'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("📚 Process Documents"):
                with st.spinner("Processing documents..."):
                    # Process uploaded files
                    processed_docs = st.session_state.document_processor.process_uploaded_files(uploaded_files)
                    
                    # Build RAG index
                    if st.session_state.rag_system.build_index(processed_docs):
                        st.success(f"✅ Successfully processed {len(processed_docs)} documents!")
                        
                        # Show document summary
                        st.subheader("📋 Document Summary")
                        for doc in processed_docs:
                            st.write(f"**{doc['filename']}** ({doc['type']}) - {doc['size']} bytes")
                    else:
                        st.error("❌ Failed to build document index")
        
        # RAG Query
        if hasattr(st.session_state.rag_system, 'index') and st.session_state.rag_system.index is not None:
            st.subheader("🔍 Query Documents")
            rag_query = st.text_input("Search in uploaded documents:")
            
            if rag_query:
                with st.spinner("Searching documents..."):
                    context = st.session_state.rag_system.get_context_for_query(rag_query)
                    
                    if context:
                        # Generate AI response with context
                        model = st.session_state.api_manager.get_working_model()
                        if model:
                            try:
                                prompt = f"""Based on the following context from the user's documents, answer the question:

CONTEXT:
{context}

QUESTION: {rag_query}

Provide a comprehensive answer based on the context provided."""

                                response = model.generate_content(prompt)
                                st.markdown("### 🤖 AI Response:")
                                st.markdown(response.text)
                                
                                st.markdown("### 📚 Source Context:")
                                st.text_area("", context, height=200)
                            except Exception as e:
                                st.error(f"Failed to generate response: {e}")
                        else:
                            st.markdown("### 📚 Found Context:")
                            st.text_area("", context, height=200)
                    else:
                        st.warning("No relevant information found in documents.")
    
    with tab3:
        st.header("🔧 Equipment Monitoring Dashboard")
        
        # Real-time alert simulation
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🚨 Simulate Alert"):
                alert = st.session_state.maintenance_pipeline.simulate_real_time_alert()
                st.session_state['current_alert'] = alert
        
        # Display current alert
        if 'current_alert' in st.session_state:
            alert = st.session_state['current_alert']
            
            # Alert severity color coding
            severity_colors = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }
            
            st.markdown(f"""
            ### {severity_colors.get(alert['severity'], '⚪')} {alert['alert_type']} Alert
            
            **Device:** {alert['device_id']} ({alert['device_type']})
            **Location:** {alert['location']}
            **Severity:** {alert['severity']}
            **Timestamp:** {alert['timestamp']}
            **Failure Probability:** {alert['failure_probability']:.2%}
            **Estimated Cost:** ${alert['estimated_cost']:,}
            **Recommended Action:** {alert['recommended_action']}
            **Urgency:** Address within {alert['urgency_hours']} hours
            """)
        
        # Equipment dashboard
        dashboard = st.session_state.maintenance_pipeline.get_maintenance_dashboard()
        
        st.subheader("📊 Equipment Status Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Equipment", dashboard['total_equipment'])
        with col2:
            st.metric("High Risk", dashboard['high_risk'], delta=-1 if dashboard['high_risk'] > 0 else 0)
        with col3:
            st.metric("Medium Risk", dashboard['medium_risk'])
        with col4:
            st.metric("Low Risk", dashboard['low_risk'], delta=1)
        
        # Equipment details
        if st.checkbox("Show Equipment Details"):
            equipment_df = pd.DataFrame.from_dict(dashboard['equipment_details'], orient='index')
            st.dataframe(equipment_df, use_container_width=True)
    
    with tab4:
        st.header("📊 System Analytics")
        
        # API usage analytics
        st.subheader("🔑 API Key Usage")
        api_usage = quota_status.get('api_key_usage', {})
        if api_usage:
            usage_df = pd.DataFrame(list(api_usage.items()), columns=['API Key', 'Requests'])
            usage_df['API Key'] = usage_df['API Key'].apply(lambda x: f"Key {x+1}")
            st.bar_chart(usage_df.set_index('API Key'))
        
        # Equipment risk distribution
        st.subheader("🔧 Equipment Risk Distribution")
        risk_data = {
            'Risk Level': ['High', 'Medium', 'Low'],
            'Count': [dashboard['high_risk'], dashboard['medium_risk'], dashboard['low_risk']]
        }
        risk_df = pd.DataFrame(risk_data)
        st.bar_chart(risk_df.set_index('Risk Level'))
        
        # Query classification stats
        if st.session_state.chat_history:
            st.subheader("🤖 Query Analysis")
            classifications = []
            for chat in st.session_state.chat_history:
                classification = st.session_state.ai_engine.query_classifier.classify_query(chat['query'])
                classifications.append(classification['category'])
            
            from collections import Counter
            category_counts = Counter(classifications)
            category_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
            st.bar_chart(category_df.set_index('Category'))
    
    with tab5:
        st.header("⚙️ System Settings")
        
        # API Management
        st.subheader("🔑 API Key Management")
        api_status = st.session_state.api_manager.get_api_status()
        
        for i, status in api_status['key_status'].items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"API Key {i+1}: {'🟢 Active' if status['active'] else '🔴 Inactive'}")
            with col2:
                st.text(f"Errors: {status['error_count']}")
            with col3:
                if st.button(f"Reset {i+1}", key=f"reset_{i}"):
                    st.session_state.api_manager.reset_key_status(i)
        
        # Quota Settings
        st.subheader("📊 Quota Configuration")
        quota_config.daily_limit = st.number_input("Daily Request Limit", 10, 1000, quota_config.daily_limit)
        quota_config.requests_per_hour = st.number_input("Hourly Request Limit", 5, 100, quota_config.requests_per_hour)
        quota_config.use_fallback_on_limit = st.checkbox("Use Fallback on Limit", quota_config.use_fallback_on_limit)
        
        # System Information
        st.subheader("ℹ️ System Information")
        st.json({
            "RAG System": "Initialized" if st.session_state.rag_system.is_initialized else "Not Initialized",
            "Documents Indexed": len(st.session_state.rag_system.documents),
            "Total Chunks": len(st.session_state.rag_system.document_chunks),
            "Equipment Monitored": len(st.session_state.maintenance_pipeline.equipment_list),
            "Chat History": len(st.session_state.chat_history) if 'chat_history' in st.session_state else 0
        })
        
        # Clear Data
        st.subheader("🗑️ Data Management")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
        with col2:
            if st.button("Clear Documents", type="secondary"):
                st.session_state.rag_system = EnhancedRAGSystem()
                st.success("Documents cleared!")
        with col3:
            if st.button("Reset System", type="secondary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Versatile AI Assistant v2.0 | Built with Streamlit & Google Gemini API
        <br>Features: Multi-API Management, RAG, Equipment Diagnostics, Real-time Monitoring
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
