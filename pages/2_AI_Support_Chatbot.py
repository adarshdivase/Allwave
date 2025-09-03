import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any
import PyPDF2
import fitz  # PyMuPDF
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
import google.generativeai as genai

# --- Basic Configuration ---
warnings.filterwarnings("ignore")

# --- Enhanced RAG Configuration ---
@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 5
    similarity_threshold: float = 1.5 # L2 distance threshold; lower is more similar. 1.5 is permissive.

config = RAGConfig()

# --- Predictive Maintenance Integration ---
@dataclass
class MaintenanceConfig:
    """Configuration for predictive maintenance system"""
    risk_threshold_high: float = 0.7
    risk_threshold_medium: float = 0.4

maintenance_config = MaintenanceConfig()

class MaintenancePipeline:
    """Pipeline connecting chatbot with a simulated predictive maintenance system"""
    def __init__(self):
        self.equipment_profiles = {
            'HVAC': {'expected_life': 15 * 365, 'maintenance_interval': 90},
            'IT_EQUIPMENT': {'expected_life': 5 * 365, 'maintenance_interval': 180},
            'ELECTRICAL': {'expected_life': 20 * 365, 'maintenance_interval': 365},
            'FIRE_SAFETY': {'expected_life': 10 * 365, 'maintenance_interval': 180}
        }
        self.maintenance_data = self._load_maintenance_data()

    def _load_maintenance_data(self) -> Dict:
        """Generates realistic-looking maintenance data for the demo"""
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'AV_EQUIPMENT']
        
        for i in range(50):
            equipment_id = f"{random.choice(equipment_types)}_{str(i+1).zfill(3)}"
            equipment_type = equipment_id.split('_')[0]
            age_days = random.randint(30, self.equipment_profiles.get(equipment_type, {}).get('expected_life', 7000))
            days_since_maintenance = random.randint(5, 300)
            
            # Simplified failure probability calculation
            age_factor = age_days / self.equipment_profiles.get(equipment_type, {}).get('expected_life', 7000)
            maintenance_factor = days_since_maintenance / self.equipment_profiles.get(equipment_type, {}).get('maintenance_interval', 365)
            failure_probability = min(0.1 + age_factor * 0.5 + maintenance_factor * 0.4, 0.99)
            
            risk_level = 'LOW'
            if failure_probability >= maintenance_config.risk_threshold_high:
                risk_level = 'HIGH'
            elif failure_probability >= maintenance_config.risk_threshold_medium:
                risk_level = 'MEDIUM'
            
            equipment_data[equipment_id] = {
                'type': equipment_type,
                'location': random.choice(['Building A', 'Building B', 'Server Room', 'Main Hall']),
                'failure_probability': failure_probability,
                'risk_level': risk_level,
                'last_maintenance': (datetime.now() - timedelta(days=days_since_maintenance)).strftime('%Y-%m-%d'),
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 5000),
            }
        return equipment_data

    def get_equipment_by_type(self, equipment_type: str) -> List[Dict]:
        return [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() if edata['type'] == equipment_type.upper()]

    def get_equipment_by_risk(self, risk_level: str) -> List[Dict]:
        return [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() if edata['risk_level'] == risk_level.upper()]

    def get_maintenance_schedule(self, days_ahead: int = 30) -> List[Dict]:
        target_date = datetime.now() + timedelta(days=days_ahead)
        items = [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() if datetime.strptime(edata['next_maintenance'], '%Y-%m-%d') <= target_date]
        return sorted(items, key=lambda x: x['next_maintenance'])
        
    def generate_maintenance_recommendations(self, equipment_data: Dict) -> List[str]:
        recommendations = []
        if equipment_data.get('risk_level') == 'HIGH':
            recommendations.extend(["🚨 URGENT: Schedule immediate inspection.", "📋 Create detailed maintenance plan."])
        elif equipment_data.get('risk_level') == 'MEDIUM':
            recommendations.extend(["⚠️ Schedule preventive maintenance soon.", "📊 Increase monitoring frequency."])
        return recommendations

# --- LLM and Response Generation ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please check your API key in secrets.toml: {e}")
    GEMINI_MODEL = None

def detect_query_intent(query: str) -> Dict[str, Any]:
    """Uses simple keyword matching to detect user intent for context retrieval"""
    query_lower = query.lower()
    intents = {
        'risk': ['risk', 'failure', 'predict', 'probability', 'alert', 'warning'],
        'schedule': ['schedule', 'calendar', 'upcoming', 'planned', 'when'],
        'status': ['status', 'health', 'condition', 'state', 'check'],
        'recommendation': ['recommend', 'suggest', 'advice', 'should'],
    }
    categories = {
        'hvac': ['hvac', 'air', 'conditioning', 'heating'],
        'it_equipment': ['it', 'computer', 'server', 'network'],
        'electrical': ['electrical', 'power', 'circuit'],
        'fire_safety': ['fire', 'smoke', 'alarm', 'sprinkler'],
    }
    
    detected_intent = next((intent for intent, keywords in intents.items() if any(kw in query_lower for kw in keywords)), None)
    detected_category = next((cat for cat, keywords in categories.items() if any(kw in query_lower for kw in keywords)), None)
    
    return {'maintenance_intent': detected_intent, 'category': detected_category}

def get_maintenance_context(query: str, maintenance_pipeline: MaintenancePipeline) -> str:
    """Retrieves and formats maintenance data as text context for the LLM"""
    query_info = detect_query_intent(query)
    intent = query_info.get('maintenance_intent')
    context_parts = []

    if intent == 'risk':
        high_risk = maintenance_pipeline.get_equipment_by_risk('HIGH')
        if high_risk:
            context_parts.append("High-Risk Equipment Data:")
            for eq in high_risk[:5]:
                context_parts.append(f"- ID: {eq['id']}, Type: {eq['type']}, Location: {eq['location']}, Failure Probability: {eq['failure_probability']:.1%}")
    elif intent == 'schedule':
        schedule = maintenance_pipeline.get_maintenance_schedule(30)
        if schedule:
            context_parts.append("Upcoming Maintenance Schedule (Next 30 Days):")
            for item in schedule[:8]:
                context_parts.append(f"- ID: {item['id']}, Date: {item['date']}, Location: {item['location']}, Cost: ${item['maintenance_cost']}")
    elif intent == 'status' and query_info.get('category'):
        equipment_list = maintenance_pipeline.get_equipment_by_type(query_info['category'])
        if equipment_list:
            context_parts.append(f"Status for {query_info['category'].replace('_', ' ').title()} Equipment:")
            for eq in equipment_list[:6]:
                context_parts.append(f"- ID: {eq['id']}, Location: {eq['location']}, Risk: {eq['risk_level']}, Last Service: {eq['last_maintenance']}")
    elif intent == 'recommendation':
        high_risk = maintenance_pipeline.get_equipment_by_risk('HIGH')
        if high_risk:
            context_parts.append("Recommendations for High-Risk Equipment:")
            for eq in high_risk[:3]:
                recs = maintenance_pipeline.generate_maintenance_recommendations(eq)
                context_parts.append(f"For {eq['id']} ({eq['type']}): {', '.join(recs)}")
    
    return "\n".join(context_parts)

def generate_llm_response(query: str, search_results: List[Dict], maintenance_pipeline: MaintenancePipeline) -> str:
    """Generates a response using the Gemini LLM, backed by retrieved context"""
    if not GEMINI_MODEL:
        return "The AI model is not configured. Please check your API key."
    
    doc_context = ""
    if search_results:
        doc_context += "Relevant information from documents:\n"
        for result in search_results[:3]:
            source = os.path.basename(result['source'])
            content_preview = result['content'].strip().replace('\n', ' ')
            doc_context += f"- From '{source}': \"{content_preview}\"\n"

    maintenance_context = get_maintenance_context(query, maintenance_pipeline)

    if not doc_context and not maintenance_context:
        high_risk_count = len(maintenance_pipeline.get_equipment_by_risk('HIGH'))
        upcoming_maintenance = len(maintenance_pipeline.get_maintenance_schedule(7))
        return f"""🤔 I couldn't find any specific information for "{query}".

🔍 **You can ask me about:**
- The status of equipment like 'HVAC' or 'IT equipment'.
- The maintenance schedule or high-risk items.
- Information from your uploaded documents (e.g., "troubleshooting guide for cameras").

📊 **Current System Status:**
- 🚨 High risk equipment: {high_risk_count}
- 📅 Maintenance due this week: {upcoming_maintenance}
"""

    prompt = f"""
You are an expert AI assistant for a facilities management team. Your goal is to provide clear, concise, and helpful answers based ONLY on the context provided below.

**User's Question:** "{query}"

---
**Context from Documents:**
{doc_context if doc_context else "No relevant document context found."}
---
**Context from Predictive Maintenance System:**
{maintenance_context if maintenance_context else "No relevant maintenance data found for this specific query."}
---

Based on all the context above, provide a direct and helpful answer. If the context is insufficient, state that you cannot find the answer in the available data. Synthesize information from both sources if applicable.
"""
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the AI response: {e}")
        return "Sorry, I encountered an issue while trying to answer your question."

# --- Document Processing and Search ---
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    return text

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

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.warning(f"Could not read {os.path.basename(file_path)} with PyMuPDF, trying PyPDF2. Error: {e}")
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(page.extract_text() for page in reader.pages)
        except Exception as e2:
            st.error(f"Failed to read PDF {os.path.basename(file_path)} with both methods. Error: {e2}")
            return ""

@st.cache_resource
def load_and_process_documents():
    """Loads all supported documents, extracts text, and returns them."""
    docs, file_paths, metadata = [], [], []
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf"]
    all_files = [f for pattern in file_patterns for f in glob.glob(pattern, recursive=True)]

    progress_bar = st.progress(0, text="Loading documents...")
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        progress_bar.progress((i + 1) / len(all_files), text=f"Processing: {file_name}")
        content = ""
        if file_path.endswith('.pdf'):
            content = extract_text_from_pdf(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            except Exception:
                with open(file_path, 'r', encoding='latin-1') as f: content = f.read()
        
        if content and len(content.strip()) > 50:
            docs.append(clean_text(content))
            file_paths.append(file_path)
            metadata.append({'name': file_name, 'path': file_path})
    
    progress_bar.empty()
    if docs: st.success(f"Successfully loaded and processed {len(docs)} documents!")
    return docs, file_paths, metadata

@st.cache_resource
def create_search_index(_documents, _file_paths, _metadata):
    """Creates a FAISS vector search index from the document contents."""
    st.info("🧠 Building semantic search index... This might take a moment.")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    all_chunks, chunk_metadata = [], []
    for i, doc in enumerate(_documents):
        chunks = smart_chunking(doc, config.chunk_size)
        all_chunks.extend(chunks)
        chunk_metadata.extend([{'source_index': i, 'path': _file_paths[i]}] * len(chunks))
            
    if not all_chunks:
        st.error("No content could be chunked for indexing.")
        return None, None, [], []

    with st.spinner("Embedding documents for semantic search..."):
        embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(embeddings.shape[1]) # Using Inner Product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    st.success(f"✅ Search index created with {len(all_chunks)} text chunks.")
    return index, model, all_chunks, chunk_metadata

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    """Performs a semantic search against the FAISS index."""
    if not query or index is None: return []
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and distances[0][i] > config.similarity_threshold:
             results.append({'content': chunks[idx], 'source': metadata[idx]['path'], 'similarity': distances[0][i]})
    return sorted(results, key=lambda x: x['similarity'], reverse=True)


# --- Main Streamlit Application ---
st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Support Chatbot with Predictive Maintenance")
st.markdown("This chatbot uses a **Retrieval-Augmented Generation (RAG)** system to answer questions based on your documents and a **Predictive Maintenance** system to provide real-time equipment health analysis.")

# --- Initialization ---
try:
    maintenance_pipeline = MaintenancePipeline()
    docs, paths, meta = load_and_process_documents()
    if docs:
        search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths, meta)
    else:
        search_index, model, all_chunks, chunk_meta = None, None, [], []
        st.warning("No documents found. The chatbot will only use maintenance data.")
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    search_index, maintenance_pipeline = None, None

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today? Ask about equipment, documents, or maintenance schedules."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your equipment or documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            search_results = search_documents(prompt, search_index, model, all_chunks, chunk_meta)
            response = generate_llm_response(prompt, search_results, maintenance_pipeline)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
