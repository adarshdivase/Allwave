import streamlit as st
import pandas as pd
import os
import io
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from collections import defaultdict
import requests # --- NEW ---

warnings.filterwarnings("ignore")

# --- Configuration ---
# --- NEW: GitHub Repository Configuration ---
# IMPORTANT: Replace this with the URL of your public GitHub repository.
# The format should be: "https://github.com/owner/repo-name"
# The code will look for files in a folder named 'data' within this repo.
GITHUB_REPO_URL = "https://github.com/streamlit/streamlit-example" # Replace with your repo URL
# --- End of New Section ---

class EnhancedRAGConfig:
    def __init__(self):
        self.chunk_size = 800
        self.chunk_overlap = 100
        self.top_k_retrieval = 8

config = EnhancedRAGConfig()

# --- Floor Plan and Schema Parser (No changes) ---
class FloorPlanAnalyzer:
    """Analyzes floor plans and building schemas for spatial intelligence"""
    def __init__(self):
        self.room_mappings = {}
        self.equipment_locations = {}
        self.load_floor_plan_data()

    def load_floor_plan_data(self):
        self.room_mappings = {
            'GYM': {'type': 'fitness', 'equipment': ['leg_press', 'fitness_equipment'], 'floor': 'ground'},
            'KITCHEN': {'type': 'food_service', 'equipment': ['cooking_equipment', 'refrigeration'], 'floor': 'ground'},
            'EXECUTIVE_DINING_1': {'type': 'meeting', 'equipment': ['av_systems', 'displays'], 'floor': 'ground'},
            'TRAINING_ROOM_6': {'type': 'training', 'equipment': ['projectors', 'audio_systems', 'cameras'], 'floor': 'second'},
        }
        self.equipment_locations = { 'A2315': 'TRAINING_ROOMS' }

    def get_room_context(self, room_name: str) -> Optional[Dict]:
        room_name_upper = room_name.upper()
        for room, details in self.room_mappings.items():
            if room in room_name_upper or room_name_upper in room:
                return {'room': room, **details}
        return None

    def find_equipment_location(self, equipment_id: str) -> Optional[str]:
        return self.equipment_locations.get(equipment_id.upper())

# --- Intelligent Query Processor (No changes) ---
class IntelligentQueryProcessor:
    def __init__(self, floor_plan_analyzer: FloorPlanAnalyzer):
        self.floor_plan = floor_plan_analyzer

    def analyze_query_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        intents = {
            'equipment_search': {'keywords': ['camera', 'projector', 'display', 'audio'], 'patterns': [r'find.*(equipment|device)'], 'confidence': 0},
            'location_query': {'keywords': ['where', 'location', 'room', 'floor'], 'patterns': [r'where is'], 'confidence': 0},
        }
        for name, data in intents.items():
            data['confidence'] += sum(1 for kw in data['keywords'] if kw in query_lower) * 0.4
            data['confidence'] += sum(1 for pat in data['patterns'] if re.search(pat, query_lower)) * 0.6
        
        primary_intent = max(intents, key=lambda k: intents[k]['confidence']) if any(d['confidence'] > 0 for d in intents.values()) else 'general_query'
        
        entities = self.extract_entities(query)
        return {'primary_intent': primary_intent, 'entities': entities}

    def extract_entities(self, query: str) -> Dict:
        entities = defaultdict(list)
        if 'camera' in query.lower(): entities['equipment_types'].append('cameras')
        for room in self.floor_plan.room_mappings.keys():
            if room.lower() in query.lower(): entities['rooms'].append(room)
        entities['equipment_ids'] = re.findall(r'\bA\d{4}[A-Z]?\b', query.upper())
        return dict(entities)

# --- Logical Response Generator (Minor changes for clarity) ---
class LogicalResponseGenerator:
    def __init__(self, floor_plan_analyzer: FloorPlanAnalyzer):
        self.floor_plan = floor_plan_analyzer

    def generate_intelligent_response(self, query_analysis: Dict, search_results: List[Dict]) -> str:
        intent = query_analysis['primary_intent']
        entities = query_analysis['entities']
        if intent == 'location_query':
            return self._handle_location_query(entities)
        else:
            return self._handle_general_query(search_results)

    def _handle_location_query(self, entities: Dict) -> str:
        response = "**Location Information**\n\n"
        equipment_ids = entities.get('equipment_ids', [])
        if not equipment_ids:
            return "Please provide an equipment ID to locate."
        for eq_id in equipment_ids:
            location = self.floor_plan.find_equipment_location(eq_id)
            if location:
                response += f"- Equipment **{eq_id}** is located in the **{location}** area.\n"
            else:
                response += f"- Could not find a location for equipment **{eq_id}**.\n"
        return response

    def _handle_general_query(self, search_results: List[Dict]) -> str:
        if not search_results:
            return "I could not find any relevant information in the provided documents."
        response = "**Found Relevant Information:**\n\n"
        for i, result in enumerate(search_results[:3], 1):
            response += f"**{i}. From: {os.path.basename(result['source'])}**\n"
            response += f"> {result['content'][:300].strip()}...\n\n"
        return response

# --- Helper Functions for Document Processing and Search ---

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- NEW: Function to get file content from a GitHub repo ---
def get_github_files(repo_url: str, folder: str = 'data') -> Optional[List[Dict]]:
    """Fetches files from a folder in a public GitHub repository."""
    try:
        owner, repo = repo_url.split('/')[-2:]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder}"
        response = requests.get(api_url)
        response.raise_for_status()
        
        files_data = []
        for file_info in response.json():
            if file_info['type'] == 'file':
                file_content_response = requests.get(file_info['download_url'])
                file_content_response.raise_for_status()
                files_data.append({
                    "name": file_info['name'],
                    "content": file_content_response.content
                })
        st.info(f"Successfully fetched {len(files_data)} files from GitHub repo.")
        return files_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching files from GitHub: {e}. Please check the repository URL and ensure it's public.")
        return None

# --- MODIFIED: Now handles different file types from raw content ---
def extract_text_from_content(file_name: str, file_content: bytes) -> str:
    """Extracts text from PDF, TXT, or CSV file content."""
    text = ""
    try:
        if file_name.lower().endswith(".pdf"):
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
        elif file_name.lower().endswith(".txt"):
            text = file_content.decode("utf-8")
        elif file_name.lower().endswith(".csv"):
            # Convert each row of the CSV into a sentence
            df = pd.read_csv(io.BytesIO(file_content))
            sentences = []
            for index, row in df.iterrows():
                row_str = ", ".join([f"{col} is {val}" for col, val in row.dropna().items()])
                sentences.append(f"Record {index+1}: {row_str}.")
            text = " ".join(sentences)
        return text
    except Exception as e:
        st.warning(f"Could not read {file_name}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    if not words: return []
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - chunk_overlap)]
    return chunks

def search_documents(query: str, model, index, chunks: List[str], metadata: List[Dict], k: int) -> List[Dict]:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [{"content": chunks[i], "source": metadata[i]['source']} for i in indices[0]]

# --- NEW: Automated setup function using @st.cache_resource ---
@st.cache_resource
def setup_and_index_github_repo(repo_url: str, model):
    """Downloads, processes, and indexes files from GitHub. Cached after first run."""
    st.write("First-time setup: Fetching and processing documents from GitHub...")
    files = get_github_files(repo_url)
    if not files:
        st.error("Setup failed: Could not retrieve files from GitHub.")
        return None, None, None

    all_chunks = []
    metadata = []
    for file in files:
        text = extract_text_from_content(file['name'], file['content'])
        if text:
            chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
            all_chunks.extend(chunks)
            metadata.extend([{'source': file['name']}] * len(chunks))

    if not all_chunks:
        st.error("Setup failed: No text could be extracted from the repository files.")
        return None, None, None
    
    with st.spinner("Creating vector embeddings... This may take a moment."):
        embeddings = model.encode(all_chunks, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
    
    st.success(f"Setup complete! Indexed {len(all_chunks)} text chunks.")
    return index, all_chunks, metadata

# --- Main Application ---
def main():
    st.set_page_config(page_title="Intelligent AI Support System", page_icon="🧠", layout="wide")
    st.title("🧠 Intelligent AI Support System")
    st.subheader("Advanced Equipment Management with Spatial Intelligence")
    
    model = load_model()

    # --- MODIFIED: Automated setup instead of manual upload ---
    # This runs once and the result is cached
    index, chunks, metadata = setup_and_index_github_repo(GITHUB_REPO_URL, model)
    
    # Initialize session state
    if 'enhanced_chatbot' not in st.session_state:
        floor_plan = FloorPlanAnalyzer()
        st.session_state.enhanced_chatbot = {
            'floor_plan': floor_plan,
            'query_processor': IntelligentQueryProcessor(floor_plan),
            'response_generator': LogicalResponseGenerator(floor_plan)
        }
        st.session_state.chat_history = []
    
    # --- MODIFIED: Removed the sidebar file uploader ---
    with st.sidebar:
        st.header("🏢 Facility Information")
        floor_plan = st.session_state.enhanced_chatbot['floor_plan']
        for room, details in floor_plan.room_mappings.items():
            st.write(f"- **{room}** ({details['type'].title()})")
        
        st.header("⚡ Quick Queries")
        # Quick queries now submit a form to trigger a rerun
        with st.form("quick_queries_form"):
             if st.form_submit_button("📷 Find all cameras"):
                 st.session_state.quick_query = "Show me all cameras"
             if st.form_submit_button("📍 Locate Equipment A2315"):
                 st.session_state.quick_query = "Where is equipment A2315 located?"

    # Main chat interface
    if st.session_state.get("quick_query"):
        query = st.session_state.pop("quick_query")
    else:
        query = st.chat_input("Ask me about equipment or locations...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # --- MODIFIED: Check if indexing was successful ---
        if index is None or not chunks:
            response = "I'm sorry, the document processing failed. Please check the GitHub repository link and file formats."
        else:
            with st.spinner("🧠 Thinking..."):
                chatbot = st.session_state.enhanced_chatbot
                query_analysis = chatbot['query_processor'].analyze_query_intent(query)
                search_results = search_documents(query, model, index, chunks, metadata, k=config.top_k_retrieval)
                response = chatbot['response_generator'].generate_intelligent_response(query_analysis, search_results)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if __name__ == "__main__":
    main()
