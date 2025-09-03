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
        self.room_mappings = {
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

        self.equipment_locations = {}
        equipment_ids = [
            'A2001', 'A2002', 'A2003', 'A2004', 'A2005', 'A2006', 'A2007', 'A2008', 'A2009', 'A2010',
            'A2160', 'A2161', 'A2162', 'A2170', 'A2171', 'A2172', 'A2225', 'A2226', 'A2227', 'A2228',
            'A2315', 'A2315A', 'A2320', 'A2321', 'A2325', 'A2326', 'A2327', 'A2328', 'A2329', 'A2330'
        ]

        for eq_id in equipment_ids:
            if eq_id.startswith('A23'):
                self.equipment_locations[eq_id] = 'TRAINING_ROOMS'
            elif eq_id.startswith('A22'):
                self.equipment_locations[eq_id] = 'EXECUTIVE_DINING'
            elif eq_id.startswith('A21'):
                self.equipment_locations[eq_id] = 'MAIN_FLOOR'
            else:
                self.equipment_locations[eq_id] = 'GENERAL_AREA'

    def get_room_context(self, room_name: str) -> Optional[Dict]:
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

        intents = {
            'equipment_search': {'keywords': ['camera', 'projector', 'display', 'audio', 'microphone', 'speaker', 'laptop', 'computer'], 'patterns': [r'show.*(?:camera|projector|display)', r'find.*(?:audio|video)', r'where.*(?:equipment|device)'], 'confidence': 0},
            'location_query': {'keywords': ['where', 'location', 'room', 'floor', 'building', 'area'], 'patterns': [r'where is', r'located in', r'room.*equipment', r'floor.*equipment'], 'confidence': 0},
            'maintenance_status': {'keywords': ['maintenance', 'service', 'repair', 'status', 'condition', 'health'], 'patterns': [r'maintenance.*status', r'when.*service', r'repair.*needed'], 'confidence': 0},
            'specifications': {'keywords': ['specs', 'specification', 'model', 'brand', 'features', 'technical'], 'patterns': [r'what.*spec', r'tell me about', r'details.*equipment'], 'confidence': 0},
            'availability': {'keywords': ['available', 'free', 'book', 'schedule', 'use'], 'patterns': [r'is.*available', r'can.*use', r'book.*room'], 'confidence': 0},
            'problem_solving': {'keywords': ['problem', 'issue', 'not working', 'broken', 'fix', 'troubleshoot'], 'patterns': [r'not working', r'broken', r'problem with', r'issue.*equipment'], 'confidence': 0}
        }

        for intent_name, intent_data in intents.items():
            keyword_matches = sum(1 for keyword in intent_data['keywords'] if keyword in query_lower)
            intent_data['confidence'] += keyword_matches * 0.3
            pattern_matches = sum(1 for pattern in intent_data['patterns'] if re.search(pattern, query_lower))
            intent_data['confidence'] += pattern_matches * 0.5

        primary_intent = max(intents.keys(), key=lambda x: intents[x]['confidence']) if any(d['confidence'] > 0 for d in intents.values()) else 'general_query'

        entities = self.extract_entities(query)
        context = self.get_conversational_context(conversation_history) if conversation_history else {}

        return {
            'primary_intent': primary_intent,
            'confidence': intents.get(primary_intent, {}).get('confidence', 0),
            'entities': entities,
            'context': context,
            'requires_logical_reasoning': intents.get(primary_intent, {}).get('confidence', 0) > 0.5,
            'spatial_query': any(word in query_lower for word in ['where', 'location', 'room', 'floor'])
        }

    def extract_entities(self, query: str) -> Dict:
        """Extract relevant entities from query"""
        entities = defaultdict(list)
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

        for room in self.floor_plan.room_mappings.keys():
            if room.lower() in query.lower():
                entities['rooms'].append(room)

        eq_id_pattern = r'\bA\d{4}[A-Z]?\b'
        entities['equipment_ids'] = re.findall(eq_id_pattern, query.upper())
        return dict(entities)

    def get_conversational_context(self, history: List[Dict]) -> Dict:
        """Extract context from conversation history"""
        context = defaultdict(list)
        if not history: return dict(context)
        
        recent_messages = history[-6:]
        for message in recent_messages:
            content = message.get('content', '').lower()
            for eq_type in ['camera', 'projector', 'display', 'audio', 'microphone']:
                if eq_type in content and eq_type not in context['mentioned_equipment']:
                    context['mentioned_equipment'].append(eq_type)
            for room in self.floor_plan.room_mappings.keys():
                if room.lower() in content and room not in context['mentioned_rooms']:
                    context['mentioned_rooms'].append(room)
        return dict(context)

# --- Enhanced Response Generator ---
class LogicalResponseGenerator:
    """Generates logical, contextual responses"""
    def __init__(self, floor_plan_analyzer: FloorPlanAnalyzer):
        self.floor_plan = floor_plan_analyzer

    def generate_intelligent_response(self, query: str, query_analysis: Dict,
                                      search_results: List[Dict],
                                      maintenance_data: Optional[Dict] = None) -> str:
        intent = query_analysis['primary_intent']
        entities = query_analysis['entities']
        context = query_analysis.get('context', {})

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

    def _handle_equipment_search(self, query: str, entities: Dict, search_results: List[Dict], context: Dict) -> str:
        response = "**Equipment Search Results**\n\n"
        equipment_types = entities.get('equipment_types', [])
        rooms = entities.get('rooms', [])

        if not equipment_types and not rooms:
             return self._handle_general_query(query, search_results, context)

        if equipment_types:
            response += f"**Looking for:** {', '.join(et.title() for et in equipment_types)}\n"
        if rooms:
            response += f"**In Location:** {', '.join(rooms)}\n\n"
            room_context = self.floor_plan.get_room_context(rooms[0])
            if room_context:
                response += f"**Room Context:** {room_context['room']} ({room_context['type']}) on the {room_context['floor']} floor.\n"
                response += f"**Typical equipment for this room type:** {', '.join(room_context['typical_equipment'])}\n\n"

        relevant_results = []
        for result in search_results:
            content_lower = result['content'].lower()
            relevance_score = 0
            for eq_type in equipment_types:
                if eq_type[:-1] in content_lower:  # Match 'camera' for 'cameras'
                    relevance_score += 2
            for room in rooms:
                if room.lower() in content_lower:
                    relevance_score += 1
            if relevance_score > 0:
                result['logical_relevance'] = relevance_score
                relevant_results.append(result)

        relevant_results.sort(key=lambda x: x.get('logical_relevance', 0), reverse=True)

        if relevant_results:
            response += "**Found Relevant Information:**\n\n"
            for i, result in enumerate(relevant_results[:3], 1):
                equipment_details = self._extract_equipment_details(result['content'])
                response += f"**{i}. {equipment_details.get('name') or 'Relevant Document Snippet'}**\n"
                response += f"   - **Source:** {os.path.basename(result['source'])}\n"
                response += f"   - **Preview:** {result['content'][:200]}...\n\n"
        else:
            response += "**No specific equipment found in documents.**\n\n"
            response += "**Suggestions:**\n"
            if equipment_types:
                response += f"- Check maintenance records for '{equipment_types[0]}'.\n"
            if rooms:
                response += f"- Broaden your search to all equipment located in '{rooms[0]}'.\n"
        return response

    def _handle_location_query(self, query: str, entities: Dict, search_results: List[Dict], context: Dict) -> str:
        response = "**Location Information**\n\n"
        equipment_ids = entities.get('equipment_ids', [])
        rooms = entities.get('rooms', [])

        found = False
        if equipment_ids:
            found = True
            for eq_id in equipment_ids:
                location = self.floor_plan.find_equipment_location(eq_id)
                if location:
                    response += f"- **Equipment {eq_id}** is located in the **{location}** area.\n"
                else:
                    response += f"- Location for Equipment **{eq_id}** not found in the floor plan data.\n"
        
        if rooms:
            found = True
            for room in rooms:
                room_info = self.floor_plan.get_room_context(room)
                if room_info:
                    response += f"\n**Details for {room_info['room']}:**\n"
                    response += f"   - **Type:** {room_info['type'].title()}\n"
                    response += f"   - **Floor:** {room_info['floor'].title()}\n"
                    response += f"   - **Typical Equipment:** {', '.join(room_info['typical_equipment'])}\n"
        
        if not found:
            response += "I can't seem to identify a specific room or equipment ID in your query. Try asking 'Where is A2315?' or 'Tell me about the GYM'.\n\n"
        
        response += "\n**General Floor Plan Context:**\n"
        response += "- **Ground Floor:** Kitchen, Gym, Executive Dining, Consultation Room\n"
        response += "- **Second Floor:** Training Rooms (6, 7, 8, 10)\n"
        return response

    def _handle_maintenance_query(self, query: str, entities: Dict, maintenance_data: Optional[Dict], context: Dict) -> str:
        response = "**Maintenance Status Analysis**\n\n"
        if not maintenance_data:
            return "**Error:** Maintenance data is not available.\n"

        equipment_types = entities.get('equipment_types', [])
        if not equipment_types:
            return "Please specify an equipment type to check its maintenance status (e.g., 'camera maintenance')."

        search_type = equipment_types[0]
        # Match singular form, e.g., 'cameras' -> 'camera'
        search_type_singular = search_type[:-1] if search_type.endswith('s') else search_type

        filtered_equipment = [
            {'id': eq_id, **eq_data} 
            for eq_id, eq_data in maintenance_data.items() 
            if search_type_singular in eq_data.get('type', '').lower()
        ]

        if filtered_equipment:
            high_priority = [eq for eq in filtered_equipment if eq.get('risk_level') == 'HIGH']
            response += f"**{search_type.title()} Equipment Status:**\n"
            response += f"- **Total Items Tracked:** {len(filtered_equipment)}\n"
            response += f"- **High Priority Items:** {len(high_priority)}\n\n"

            if high_priority:
                response += "**Immediate Action Recommended:**\n"
                for eq in high_priority[:3]:
                    response += (f"- **{eq['id']}** ({eq['type']}) has a **{eq['failure_probability']:.0%} failure risk**.\n"
                                 f"  - Location: {eq['location']} | Last service: {eq['last_maintenance']}\n")
        else:
            response += f"No maintenance records found for '{search_type.title()}'.\n"
        return response

    def _extract_equipment_details(self, content: str) -> Dict:
        details = {}
        # Simple extraction, can be improved with more complex regex
        if 'camera' in content.lower(): details['type'] = 'Camera'
        elif 'projector' in content.lower(): details['type'] = 'Projector'
        elif 'microphone' in content.lower(): details['type'] = 'Audio'
        elif 'display' in content.lower(): details['type'] = 'Display'
        
        model_match = re.search(r'\b([A-Z0-9-]{4,})\b', content)
        if model_match:
            details['name'] = model_match.group(1)
        return details

    def _handle_specifications_query(self, query: str, entities: Dict, search_results: List[Dict], context: Dict) -> str:
        response = "**Technical Specifications**\n\n"
        if not search_results:
            return "I couldn't find any documents that might contain specifications for your query."

        found_specs = False
        for result in search_results[:2]:
            specs = self._extract_specifications(result['content'])
            if specs:
                found_specs = True
                response += f"**From {os.path.basename(result['source'])}:**\n"
                for key, value in specs.items():
                    response += f"- **{key}:** {value}\n"
                response += "\n"
        
        if not found_specs:
            response += "No specific technical specs found, but here is the most relevant document snippet:\n"
            response += f"> {search_results[0]['content'][:400]}...\n"

        return response

    def _extract_specifications(self, content: str) -> Dict:
        specs = {}
        spec_patterns = [
            (r'resolution[:\s]+([\w\s]+x[\w\s]+)', 'Resolution'),
            (r'brightness[:\s]+([0-9,]+\s*lumens?)', 'Brightness'),
            (r'connectivity[:\s]+([^.\n]+)', 'Connectivity'),
            (r'warranty[:\s]+([^.\n]+)', 'Warranty'),
        ]
        for pattern, key in spec_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                specs[key] = match.group(1).strip()
        return specs

    def _handle_problem_solving(self, query: str, entities: Dict, search_results: List[Dict], context: Dict) -> str:
        response = "**Problem Analysis & Suggested Solutions**\n\n"
        problem_keywords = ['not working', 'broken', 'error', 'issue', 'problem']
        identified_problems = [kw for kw in problem_keywords if kw in query.lower()]
        
        if identified_problems:
            response += f"**Identified Issue:** Problem related to '{identified_problems[0]}'.\n\n"
        
        equipment_types = entities.get('equipment_types', [])
        query_lower = query.lower()

        if 'cameras' in equipment_types or 'camera' in query_lower:
            response += "**Camera Troubleshooting Steps:**\n"
            response += "1. Ensure the camera is powered on and the lens cap is removed.\n"
            response += "2. Check all cable connections (USB, HDMI, Power).\n"
            response += "3. Restart the connected computer or video system.\n"
            response += "4. Verify the correct video input is selected in your software.\n"
            response += "5. If the problem persists, please create a support ticket.\n\n"
        else:
            response += "**General Troubleshooting Steps:**\n"
            response += "1. Check for power and ensure the device is turned on.\n"
            response += "2. Securely reconnect all relevant cables.\n"
            response += "3. Power cycle the device (turn it off and on again).\n\n"

        if search_results:
            response += "**Related Documentation Found:**\n"
            for result in search_results[:2]:
                response += f"- {os.path.basename(result['source'])}\n"
        return response

    def _handle_availability_query(self, query: str, entities: Dict, context: Dict) -> str:
        response = "**Availability & Booking Information**\n\n"
        rooms = entities.get('rooms', [])
        if rooms:
            for room in rooms:
                room_info = self.floor_plan.get_room_context(room)
                if room_info:
                    response += f"For **{room_info['room']}**:\n"
                    response += f"- This is a **{room_info['type']}** type room typically containing: {', '.join(room_info['typical_equipment'])}.\n"
        response += "\nTo check current availability or to book a room, please refer to the official company-wide booking system.\n"
        return response

    def _handle_general_query(self, query: str, search_results: List[Dict], context: Dict) -> str:
        response = "**Information Found**\n\n"
        if search_results:
            for i, result in enumerate(search_results[:3], 1):
                response += f"**{i}. From: {os.path.basename(result['source'])}**\n"
                content_preview = result['content'][:300].strip()
                response += f"> {content_preview}...\n\n"
        else:
            response += "**I couldn't find specific information for your query.**\n\n"
            response += "You can ask me about:\n"
            response += "- Equipment locations (e.g., 'Where is A2315?')\n"
            response += "- Room details (e.g., 'What's in the GYM?')\n"
            response += "- Maintenance status (e.g., 'camera maintenance')\n"
            response += "- Troubleshooting (e.g., 'The projector is not working')\n"
        return response

# --- Helper Functions for Document Processing and Search ---

@st.cache_resource
def load_model():
    """Load the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_doc(uploaded_file):
    """Extract text from PDF or TXT file."""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            doc.close()
        elif uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def process_and_index_files(uploaded_files, model, chunk_size, chunk_overlap):
    """Process uploaded files and create a FAISS index."""
    all_chunks = []
    metadata = []
    
    for file in uploaded_files:
        text = extract_text_from_doc(file)
        if text:
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            metadata.extend([{'source': file.name}] * len(chunks))

    if not all_chunks:
        st.warning("No text could be extracted from the uploaded files.")
        return None, None, None

    with st.spinner("Creating vector embeddings for the documents... This may take a moment."):
        embeddings = model.encode(all_chunks, show_progress_bar=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
    
    st.success(f"Successfully indexed {len(all_chunks)} chunks from {len(uploaded_files)} files.")
    return index, all_chunks, metadata

def search_documents(query: str, model, index, chunks: List[str], metadata: List[Dict], k: int) -> List[Dict]:
    """Search for relevant documents using FAISS."""
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        results.append({
            "content": chunks[idx],
            "source": metadata[idx]['source'],
            "score": distances[0][i]
        })
    return results

def generate_dummy_maintenance_data():
    """Creates a sample dictionary of maintenance data."""
    return {
        "A2315": {"type": "PTZ Camera", "location": "TRAINING_ROOM_6", "last_maintenance": "2025-07-15", "failure_probability": 0.85, "risk_level": "HIGH"},
        "A2320": {"type": "Ceiling Microphone", "location": "TRAINING_ROOM_7", "last_maintenance": "2025-08-01", "failure_probability": 0.20, "risk_level": "LOW"},
        "A2225": {"type": "8K Display", "location": "EXECUTIVE_DINING_1", "last_maintenance": "2025-06-20", "failure_probability": 0.92, "risk_level": "HIGH"},
        "CAM-04": {"type": "Security Camera", "location": "Lobby", "last_maintenance": "2025-08-22", "failure_probability": 0.40, "risk_level": "MEDIUM"},
    }

# --- Main Application ---

def main():
    st.set_page_config(page_title="Intelligent AI Support System", page_icon="🧠", layout="wide")
    
    st.title("🧠 Intelligent AI Support System")
    st.subheader("Advanced Equipment Management with Spatial Intelligence")
    
    # Initialize components and state
    if 'enhanced_chatbot' not in st.session_state:
        floor_plan_analyzer = FloorPlanAnalyzer()
        st.session_state.enhanced_chatbot = {
            'floor_plan': floor_plan_analyzer,
            'query_processor': IntelligentQueryProcessor(floor_plan_analyzer),
            'response_generator': LogicalResponseGenerator(floor_plan_analyzer)
        }
        st.session_state.chat_history = []
        st.session_state.search_ready = False
        st.session_state.maintenance_data = generate_dummy_maintenance_data()
    
    chatbot_components = st.session_state.enhanced_chatbot
    model = load_model()

    # --- Sidebar for File Upload and Control ---
    with st.sidebar:
        st.header("📚 Document Management")
        uploaded_files = st.file_uploader(
            "Upload technical manuals, floor plans, etc. (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("Process and Index Documents"):
            if uploaded_files:
                index, chunks, metadata = process_and_index_files(
                    uploaded_files, model, config.chunk_size, config.chunk_overlap
                )
                if index is not None:
                    st.session_state.search_index = index
                    st.session_state.chunks = chunks
                    st.session_state.chunk_metadata = metadata
                    st.session_state.search_ready = True
            else:
                st.warning("Please upload at least one document.")
        
        st.divider()
        st.header("🏢 Facility Information")
        st.subheader("📐 Floor Plan Overview")
        floor_plan = chatbot_components['floor_plan']
        for room, details in floor_plan.room_mappings.items():
            st.write(f"- **{room}** ({details['type'].title()})")
        
        st.subheader("⚡ Quick Queries")
        if st.button("📷 Find all cameras"):
            st.session_state.quick_query = "Show me all cameras in the building"
            st.rerun()
        if st.button("🏢 Room equipment"):
            st.session_state.quick_query = "What equipment is in the training rooms?"
            st.rerun()
        if st.button("📍 Equipment location"):
            st.session_state.quick_query = "Where is equipment A2315 located?"
            st.rerun()

    # --- Main Chat Interface ---
    
    # Handle quick queries
    if st.session_state.get("quick_query"):
        query = st.session_state.pop("quick_query")
        st.chat_input("Ask me anything...", disabled=True) # Disable input while processing
    else:
        query = st.chat_input("Ask me about equipment, locations, or maintenance...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        if not st.session_state.search_ready:
            st.warning("Please upload and process documents first to enable search functionality.")
            response = "I'm ready to help, but you need to upload and process some documents using the sidebar first so I have information to search through."
        else:
            with st.spinner("🧠 Analyzing query and searching documents..."):
                query_analysis = chatbot_components['query_processor'].analyze_query_intent(
                    query, st.session_state.chat_history
                )
                
                search_results = search_documents(
                    query,
                    model,
                    st.session_state.search_index,
                    st.session_state.chunks,
                    st.session_state.chunk_metadata,
                    k=config.top_k_retrieval
                )
                
                response = chatbot_components['response_generator'].generate_intelligent_response(
                    query, query_analysis, search_results,
                    st.session_state.get('maintenance_data', {})
                )
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun() # Rerun to display the latest message

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
