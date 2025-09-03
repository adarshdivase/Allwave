# Enhanced AI Support Chatbot with Logical Reasoning and Schema Awareness
import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
from collections import defaultdict

warnings.filterwarnings("ignore")

# --- Enhanced Configuration ---
class EnhancedRAGConfig:
    def __init__(self):
        self.chunk_size = 800
        self.chunk_overlap = 100
        self.top_k_retrieval = 8
        self.similarity_threshold = 0.25
        self.max_context_length = 4000
        self.logical_reasoning_enabled = True
        self.schema_aware = True

config = EnhancedRAGConfig()

# --- Enhanced Floor Plan and Schema Parser ---
class FloorPlanAnalyzer:
    """Analyzes floor plans and building schemas for spatial intelligence"""
    
    def __init__(self):
        self.room_mappings = {}
        self.equipment_locations = {}
        self.spatial_relationships = {}
        self.load_floor_plan_data()
    
    def load_floor_plan_data(self):
        """Load and parse floor plan information"""
        # Parse the provided floor plan data
        self.room_mappings = {
            # From the floor plan document
            'GYM': {'type': 'fitness', 'equipment': ['leg_press', 'fitness_equipment'], 'floor': 'ground'},
            'KITCHEN': {'type': 'food_service', 'equipment': ['cooking_equipment', 'refrigeration'], 'floor': 'ground'},
            'EXECUTIVE_DINING_1': {'type': 'meeting', 'equipment': ['av_systems', 'displays'], 'floor': 'ground'},
            'EXECUTIVE_DINING_2': {'type': 'meeting', 'equipment': ['av_systems', 'displays'], 'floor': 'ground'},
            'TRAINING_ROOM_6': {'type': 'training', 'equipment': ['projectors', 'audio_systems', 'cameras'], 'floor': 'second'},
            'TRAINING_ROOM_7': {'type': 'training', 'equipment': ['projectors', 'audio_systems', 'cameras'], 'floor': 'second'},
            'TRAINING_ROOM_8': {'type': 'training', 'equipment': ['projectors', 'audio_systems', 'cameras'], 'floor': 'second'},
            'TRAINING_ROOM_10': {'type': 'training', 'equipment': ['projectors', 'audio_systems', 'cameras'], 'floor': 'second'},
            'CONSULTATION_ROOM': {'type': 'meeting', 'equipment': ['video_conferencing', 'displays'], 'floor': 'ground'},
            'STORE': {'type': 'storage', 'equipment': ['inventory_systems'], 'floor': 'ground'},
            'SERVICE': {'type': 'utility', 'equipment': ['hvac_controls', 'electrical_panels'], 'floor': 'ground'},
        }
        
        # Equipment ID mappings from floor plan
        self.equipment_locations = {}
        equipment_ids = [
            'A2001', 'A2002', 'A2003', 'A2004', 'A2005', 'A2006', 'A2007', 'A2008', 'A2009', 'A2010',
            'A2160', 'A2161', 'A2162', 'A2170', 'A2171', 'A2172', 'A2225', 'A2226', 'A2227', 'A2228',
            'A2315', 'A2315A', 'A2320', 'A2321', 'A2325', 'A2326', 'A2327', 'A2328', 'A2329', 'A2330'
        ]
        
        # Map equipment IDs to likely locations based on floor plan
        for eq_id in equipment_ids:
            if eq_id.startswith('A23'):
                self.equipment_locations[eq_id] = 'TRAINING_ROOMS'
            elif eq_id.startswith('A22'):
                self.equipment_locations[eq_id] = 'EXECUTIVE_DINING'
            elif eq_id.startswith('A21'):
                self.equipment_locations[eq_id] = 'MAIN_FLOOR'
            else:
                self.equipment_locations[eq_id] = 'GENERAL_AREA'
    
    def get_room_context(self, room_name: str) -> Dict:
        """Get contextual information about a room"""
        room_name_upper = room_name.upper()
        for room, details in self.room_mappings.items():
            if room in room_name_upper or room_name_upper in room:
                return {
                    'room': room,
                    'type': details['type'],
                    'typical_equipment': details['equipment'],
                    'floor': details['floor']
                }
        return None
    
    def find_equipment_location(self, equipment_id: str) -> Optional[str]:
        """Find the location of specific equipment"""
        return self.equipment_locations.get(equipment_id.upper())

# --- Intelligent Query Processor ---
class IntelligentQueryProcessor:
    """Advanced query processing with logical reasoning"""
    
    def __init__(self, floor_plan_analyzer: FloorPlanAnalyzer):
        self.floor_plan = floor_plan_analyzer
        self.context_memory = []
        
    def analyze_query_intent(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """Enhanced query analysis with logical reasoning"""
        query_lower = query.lower()
        
        # Intent categories with enhanced detection
        intents = {
            'equipment_search': {
                'keywords': ['camera', 'projector', 'display', 'audio', 'microphone', 'speaker', 'laptop', 'computer'],
                'patterns': [r'show.*(?:camera|projector|display)', r'find.*(?:audio|video)', r'where.*(?:equipment|device)'],
                'confidence': 0
            },
            'location_query': {
                'keywords': ['where', 'location', 'room', 'floor', 'building', 'area'],
                'patterns': [r'where is', r'located in', r'room.*equipment', r'floor.*equipment'],
                'confidence': 0
            },
            'maintenance_status': {
                'keywords': ['maintenance', 'service', 'repair', 'status', 'condition', 'health'],
                'patterns': [r'maintenance.*status', r'when.*service', r'repair.*needed'],
                'confidence': 0
            },
            'specifications': {
                'keywords': ['specs', 'specification', 'model', 'brand', 'features', 'technical'],
                'patterns': [r'what.*spec', r'tell me about', r'details.*equipment'],
                'confidence': 0
            },
            'availability': {
                'keywords': ['available', 'free', 'book', 'schedule', 'use'],
                'patterns': [r'is.*available', r'can.*use', r'book.*room'],
                'confidence': 0
            },
            'problem_solving': {
                'keywords': ['problem', 'issue', 'not working', 'broken', 'fix', 'troubleshoot'],
                'patterns': [r'not working', r'broken', r'problem with', r'issue.*equipment'],
                'confidence': 0
            }
        }
        
        # Calculate confidence scores
        for intent_name, intent_data in intents.items():
            # Keyword matching
            keyword_matches = sum(1 for keyword in intent_data['keywords'] if keyword in query_lower)
            intent_data['confidence'] += keyword_matches * 0.3
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in intent_data['patterns'] if re.search(pattern, query_lower))
            intent_data['confidence'] += pattern_matches * 0.5
        
        primary_intent = max(intents.keys(), key=lambda x: intents[x]['confidence'])
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Determine context from conversation history
        context = self.get_conversational_context(conversation_history) if conversation_history else {}
        
        return {
            'primary_intent': primary_intent,
            'confidence': intents[primary_intent]['confidence'],
            'entities': entities,
            'context': context,
            'requires_logical_reasoning': intents[primary_intent]['confidence'] > 0.5,
            'spatial_query': any(word in query_lower for word in ['where', 'location', 'room', 'floor'])
        }
    
    def extract_entities(self, query: str) -> Dict:
        """Extract relevant entities from query"""
        entities = {
            'equipment_types': [],
            'rooms': [],
            'equipment_ids': [],
            'brands': [],
            'actions': []
        }
        
        # Equipment type extraction
        equipment_patterns = {
            'cameras': r'\b(?:camera|webcam|camcorder|video camera)\b',
            'displays': r'\b(?:display|monitor|screen|tv|television|projector)\b',
            'audio': r'\b(?:audio|microphone|mic|speaker|sound|headphone)\b',
            'computers': r'\b(?:laptop|computer|pc|workstation|desktop)\b',
            'network': r'\b(?:network|wifi|router|switch|cable)\b'
        }
        
        for eq_type, pattern in equipment_patterns.items():
            if re.search(pattern, query.lower()):
                entities['equipment_types'].append(eq_type)
        
        # Room extraction
        for room in self.floor_plan.room_mappings.keys():
            if room.lower() in query.lower():
                entities['rooms'].append(room)
        
        # Equipment ID extraction (A-series IDs from floor plan)
        eq_id_pattern = r'\bA\d{4}[A-Z]?\b'
        entities['equipment_ids'] = re.findall(eq_id_pattern, query.upper())
        
        return entities
    
    def get_conversational_context(self, history: List[Dict]) -> Dict:
        """Extract context from conversation history"""
        context = {
            'previous_topics': [],
            'mentioned_equipment': [],
            'mentioned_rooms': []
        }
        
        if not history:
            return context
            
        # Analyze last 3 messages for context
        recent_messages = history[-6:]  # User and assistant messages
        
        for message in recent_messages:
            content = message.get('content', '').lower()
            
            # Extract previously mentioned equipment
            for eq_type in ['camera', 'projector', 'display', 'audio', 'microphone']:
                if eq_type in content:
                    context['mentioned_equipment'].append(eq_type)
            
            # Extract previously mentioned rooms
            for room in self.floor_plan.room_mappings.keys():
                if room.lower() in content:
                    context['mentioned_rooms'].append(room)
        
        return context

# --- Enhanced Response Generator ---
class LogicalResponseGenerator:
    """Generates logical, contextual responses"""
    
    def __init__(self, floor_plan_analyzer: FloorPlanAnalyzer):
        self.floor_plan = floor_plan_analyzer
        
    def generate_intelligent_response(self, query: str, query_analysis: Dict, 
                                    search_results: List[Dict], 
                                    maintenance_data: Dict = None) -> str:
        """Generate logical, contextual responses"""
        
        intent = query_analysis['primary_intent']
        entities = query_analysis['entities']
        context = query_analysis.get('context', {})
        
        # Route to appropriate response generator
        if intent == 'equipment_search':
            return self._handle_equipment_search(query, entities, search_results, context)
        elif intent == 'location_query':
            return self._handle_location_query(query, entities, search_results, context)
        elif intent == 'maintenance_status':
            return self._handle_maintenance_query(query, entities, maintenance_data, context)
        elif intent == 'specifications':
            return self._handle_specifications_query(query, entities, search_results, context)
        elif intent == 'problem_solving':
            return self._handle_problem_solving(query, entities, search_results, context)
        elif intent == 'availability':
            return self._handle_availability_query(query, entities, context)
        else:
            return self._handle_general_query(query, search_results, context)
    
    def _handle_equipment_search(self, query: str, entities: Dict, 
                               search_results: List[Dict], context: Dict) -> str:
        """Handle equipment search queries with logical reasoning"""
        
        response = "**Equipment Search Results**\n\n"
        
        equipment_types = entities.get('equipment_types', [])
        rooms = entities.get('rooms', [])
        
        if equipment_types:
            response += f"**Looking for:** {', '.join(equipment_types).title()}\n\n"
            
            # Logical reasoning based on room context
            if rooms:
                room_context = self.floor_plan.get_room_context(rooms[0])
                if room_context:
                    response += f"**Room Context:** {room_context['room']} ({room_context['type']})\n"
                    response += f"**Typical equipment for this room type:** {', '.join(room_context['typical_equipment'])}\n\n"
        
        # Process search results with logical filtering
        relevant_results = []
        for result in search_results:
            content = result['content'].lower()
            
            # Score relevance based on entity matches
            relevance_score = 0
            for eq_type in equipment_types:
                if eq_type in content:
                    relevance_score += 2
            
            # Boost score for room matches
            for room in rooms:
                if room.lower() in content:
                    relevance_score += 1
            
            if relevance_score > 0:
                result['logical_relevance'] = relevance_score
                relevant_results.append(result)
        
        # Sort by logical relevance
        relevant_results.sort(key=lambda x: x.get('logical_relevance', 0), reverse=True)
        
        if relevant_results:
            response += "**Found Equipment:**\n\n"
            for i, result in enumerate(relevant_results[:3], 1):
                # Extract specific equipment details
                equipment_details = self._extract_equipment_details(result['content'])
                response += f"**{i}. {equipment_details['name'] or 'Equipment Item'}**\n"
                response += f"   • **Type:** {equipment_details['type'] or 'Not specified'}\n"
                response += f"   • **Location:** {equipment_details['location'] or 'See details'}\n"
                response += f"   • **Source:** {os.path.basename(result['source'])}\n\n"
        else:
            response += "**No specific equipment found in documents.**\n\n"
            # Provide logical alternatives
            response += "**Suggestions:**\n"
            if equipment_types:
                response += f"• Check maintenance records for {equipment_types[0]} equipment\n"
                response += f"• Look in rooms typically containing {equipment_types[0]} equipment\n"
            if rooms:
                response += f"• Browse all equipment in {rooms[0]}\n"
        
        return response
    
    def _handle_location_query(self, query: str, entities: Dict, 
                             search_results: List[Dict], context: Dict) -> str:
        """Handle location-based queries with spatial reasoning"""
        
        response = "**Location Information**\n\n"
        
        equipment_ids = entities.get('equipment_ids', [])
        rooms = entities.get('rooms', [])
        
        # Handle specific equipment ID queries
        if equipment_ids:
            for eq_id in equipment_ids:
                location = self.floor_plan.find_equipment_location(eq_id)
                if location:
                    response += f"**Equipment {eq_id}:**\n"
                    response += f"   • **Area:** {location}\n"
                    response += f"   • **Floor:** Second Floor (based on floor plan)\n\n"
                else:
                    response += f"**Equipment {eq_id}:** Location not found in floor plan\n\n"
        
        # Handle room-based queries
        if rooms:
            for room in rooms:
                room_info = self.floor_plan.get_room_context(room)
                if room_info:
                    response += f"**{room_info['room']}:**\n"
                    response += f"   • **Type:** {room_info['type'].title()}\n"
                    response += f"   • **Floor:** {room_info['floor'].title()}\n"
                    response += f"   • **Typical Equipment:** {', '.join(room_info['typical_equipment'])}\n\n"
        
        # Add floor plan context
        response += "**Floor Plan Context:**\n"
        response += "• **Ground Floor:** Kitchen, Gym, Executive Dining, Consultation Room\n"
        response += "• **Second Floor:** Training Rooms (6, 7, 8, 10), Meeting Areas\n"
        response += "• **Equipment IDs:** A-series numbering system used throughout facility\n\n"
        
        return response
    
    def _handle_maintenance_query(self, query: str, entities: Dict, 
                                maintenance_data: Dict, context: Dict) -> str:
        """Handle maintenance queries with predictive logic"""
        
        response = "**Maintenance Status Analysis**\n\n"
        
        if not maintenance_data:
            return "**Error:** Maintenance system not available\n\n"
        
        # Logical analysis based on query entities
        equipment_types = entities.get('equipment_types', [])
        rooms = entities.get('rooms', [])
        
        if equipment_types:
            # Filter maintenance data by equipment type
            filtered_equipment = []
            for eq_id, eq_data in maintenance_data.items():
                eq_type = eq_data.get('type', '').lower()
                for search_type in equipment_types:
                    if search_type in eq_type or any(search_type in keyword for keyword in ['camera', 'audio', 'display']):
                        filtered_equipment.append({'id': eq_id, **eq_data})
            
            if filtered_equipment:
                # Logical prioritization
                high_priority = [eq for eq in filtered_equipment if eq['risk_level'] == 'HIGH']
                
                response += f"**{equipment_types[0].title()} Equipment Status:**\n"
                response += f"   • **Total Items:** {len(filtered_equipment)}\n"
                response += f"   • **High Priority:** {len(high_priority)}\n\n"
                
                if high_priority:
                    response += "**Immediate Action Required:**\n"
                    for eq in high_priority[:3]:
                        response += f"• **{eq['id']}** - {eq['failure_probability']:.1%} failure risk\n"
                        response += f"  Location: {eq['location']} | Last service: {eq['last_maintenance']}\n"
        
        return response
    
    def _extract_equipment_details(self, content: str) -> Dict:
        """Extract specific equipment details from content"""
        details = {'name': None, 'type': None, 'location': None, 'brand': None}
        
        # Extract equipment names/models
        model_patterns = [
            r'([A-Z]+[0-9]+[A-Z]*)',  # Model numbers like HD450, PTZ200
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # Brand Model combinations
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, content)
            if matches:
                details['name'] = matches[0]
                break
        
        # Extract types
        if 'camera' in content.lower():
            details['type'] = 'Camera'
        elif 'projector' in content.lower():
            details['type'] = 'Projector'
        elif 'microphone' in content.lower():
            details['type'] = 'Audio'
        elif 'display' in content.lower():
            details['type'] = 'Display'
        
        return details
    
    def _handle_specifications_query(self, query: str, entities: Dict, 
                                   search_results: List[Dict], context: Dict) -> str:
        """Handle specification queries"""
        
        response = "**Technical Specifications**\n\n"
        
        if search_results:
            for result in search_results[:2]:
                specs = self._extract_specifications(result['content'])
                if specs:
                    response += f"**From {os.path.basename(result['source'])}:**\n"
                    for key, value in specs.items():
                        response += f"   • **{key}:** {value}\n"
                    response += "\n"
        
        return response
    
    def _extract_specifications(self, content: str) -> Dict:
        """Extract technical specifications from content"""
        specs = {}
        
        # Common specification patterns
        spec_patterns = [
            (r'resolution[:\s]+([0-9]+x[0-9]+)', 'Resolution'),
            (r'brightness[:\s]+([0-9,]+\s*lumens?)', 'Brightness'),
            (r'connectivity[:\s]+([^.\n]+)', 'Connectivity'),
            (r'warranty[:\s]+([^.\n]+)', 'Warranty'),
        ]
        
        for pattern, key in spec_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                specs[key] = match.group(1).strip()
        
        return specs
    
    def _handle_problem_solving(self, query: str, entities: Dict, 
                              search_results: List[Dict], context: Dict) -> str:
        """Handle troubleshooting and problem-solving queries"""
        
        response = "**Problem Analysis & Solutions**\n\n"
        
        # Extract problem keywords
        problem_keywords = ['not working', 'broken', 'error', 'issue', 'problem', 'malfunction']
        identified_problems = [kw for kw in problem_keywords if kw in query.lower()]
        
        if identified_problems:
            response += f"**Identified Issue:** {identified_problems[0]}\n\n"
        
        # Provide logical troubleshooting steps
        equipment_types = entities.get('equipment_types', [])
        
        if 'cameras' in equipment_types or 'camera' in query.lower():
            response += "**Camera Troubleshooting Steps:**\n"
            response += "1. Check power cable connections\n"
            response += "2. Verify USB/network cable integrity\n"
            response += "3. Test camera with different software\n"
            response += "4. Check driver installation\n"
            response += "5. Contact IT support if issues persist\n\n"
        
        # Add relevant documentation
        if search_results:
            response += "**Related Documentation:**\n"
            for result in search_results[:2]:
                response += f"• {os.path.basename(result['source'])}\n"
        
        return response
    
    def _handle_availability_query(self, query: str, entities: Dict, context: Dict) -> str:
        """Handle availability and booking queries"""
        
        response = "**Availability Information**\n\n"
        
        rooms = entities.get('rooms', [])
        
        if rooms:
            for room in rooms:
                room_info = self.floor_plan.get_room_context(room)
                if room_info:
                    response += f"**{room_info['room']}:**\n"
                    response += f"   • **Type:** {room_info['type'].title()}\n"
                    response += f"   • **Equipment:** {', '.join(room_info['typical_equipment'])}\n"
                    response += f"   • **Status:** Check booking system for current availability\n\n"
        
        response += "**Booking Recommendations:**\n"
        response += "• Contact facility management for room reservations\n"
        response += "• Check internal booking system\n"
        response += "• Verify equipment requirements in advance\n\n"
        
        return response
    
    def _handle_general_query(self, query: str, search_results: List[Dict], context: Dict) -> str:
        """Handle general queries with contextual information"""
        
        response = "**Information Found**\n\n"
        
        if search_results:
            for i, result in enumerate(search_results[:3], 1):
                response += f"**{i}. {os.path.basename(result['source'])}**\n"
                content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
                response += f"{content_preview}\n\n"
        else:
            response += "**No specific information found.**\n\n"
            response += "**Available Services:**\n"
            response += "• Equipment location and specifications\n"
            response += "• Room and facility information\n"
            response += "• Maintenance status and scheduling\n"
            response += "• Technical support and troubleshooting\n\n"
        
        return response

# Update the main application to use enhanced components
def create_enhanced_chatbot():
    """Create the enhanced chatbot with logical reasoning"""
    
    # Initialize components
    floor_plan_analyzer = FloorPlanAnalyzer()
    query_processor = IntelligentQueryProcessor(floor_plan_analyzer)
    response_generator = LogicalResponseGenerator(floor_plan_analyzer)
    
    return {
        'floor_plan': floor_plan_analyzer,
        'query_processor': query_processor,
        'response_generator': response_generator
    }

# Enhanced main function with intelligent processing
def main():
    """Enhanced main Streamlit application"""
    st.set_page_config(
        page_title="Intelligent AI Support System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize enhanced chatbot
    if 'enhanced_chatbot' not in st.session_state:
        st.session_state.enhanced_chatbot = create_enhanced_chatbot()
    
    chatbot_components = st.session_state.enhanced_chatbot
    
    st.title("🧠 Intelligent AI Support System")
    st.subheader("Advanced Equipment Management with Spatial Intelligence")
    
    # Enhanced chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    query = st.chat_input("Ask me about equipment, locations, maintenance, or any technical questions...")
    
    if query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Process query with enhanced intelligence
        with st.spinner("🧠 Analyzing query and generating intelligent response..."):
            
            # Analyze query with logical reasoning
            query_analysis = chatbot_components['query_processor'].analyze_query_intent(
                query, st.session_state.chat_history
            )
            
            # Search documents (using existing search function)
            search_results = []
            if st.session_state.get('search_ready', False):
                search_results = search_documents(
                    query,
                    st.session_state.search_index,
                    st.session_state.chunks,
                    st.session_state.chunk_metadata,
                    k=8
                )
            
            # Generate intelligent response
            response = chatbot_components['response_generator'].generate_intelligent_response(
                query, query_analysis, search_results, 
                st.session_state.get('maintenance_data', {})
            )
            
            # Add debug info for development
            with st.expander("🔍 Query Analysis (Debug)", expanded=False):
                st.json(query_analysis)
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sidebar with enhanced information
    with st.sidebar:
        st.header("🏢 Facility Information")
        
        # Floor plan summary
        st.subheader("📐 Floor Plan Overview")
        floor_plan = chatbot_components['floor_plan']
        
        st.write("**Rooms Available:**")
        for room, details in floor_plan.room_mappings.items():
            st.write(f"• **{room}** ({details['type']})")
        
        st.write("**Equipment Zones:**")
        st.write("• Training Rooms: A23xx series")
        st.write("• Executive Areas: A22xx series") 
        st.write("• Main Floor: A21xx series")
        
        # Quick actions
        st.subheader("⚡ Quick Queries")
        if st.button("📷 Find all cameras"):
            st.session_state.quick_query = "Show me all cameras in the building"
        
        if st.button("🏢 Room equipment"):
            st.session_state.quick_query = "What equipment is in the training rooms?"
        
        if st.button("📍 Equipment locations"):
            st.session_state.quick_query = "Where is equipment A2315 located?"

if __name__ == "__main__":
    main()
