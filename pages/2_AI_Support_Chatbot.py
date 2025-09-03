# app.py
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
import mailbox  # Standard library for email processing
from email import policy
from email.parser import BytesParser

# --- Basic Configuration ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="AI Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced RAG Configuration ---
@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.4 # Corrected threshold

config = RAGConfig()

# --- Predictive Maintenance Integration (Simulation) ---
class MaintenancePipeline:
    def __init__(self):
        self.maintenance_data = self._load_maintenance_data()

    def _load_maintenance_data(self) -> Dict:
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY']
        for i in range(20):
            eq_type = random.choice(equipment_types)
            fail_prob = random.uniform(0.1, 0.9)
            equipment_data[f"{eq_type}_{i+1}"] = {
                'type': eq_type,
                'location': f"Building A - Floor {random.randint(1,4)}",
                'failure_probability': fail_prob,
                'risk_level': 'HIGH' if fail_prob > 0.7 else 'MEDIUM' if fail_prob > 0.4 else 'LOW',
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 5000)
            }
        return equipment_data
    
    def get_equipment_by_risk(self, risk_level: str) -> List[Dict]:
        return [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() if edata['risk_level'] == risk_level.upper()]

    def get_maintenance_schedule(self, days_ahead: int = 30) -> List[Dict]:
        target_date = datetime.now() + timedelta(days=days_ahead)
        items = [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() if datetime.strptime(edata['next_maintenance'], '%Y-%m-%d') <= target_date]
        return sorted(items, key=lambda x: x['next_maintenance'])

# --- LLM and Response Generation ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please check your API key in secrets.toml: {e}")
    GEMINI_MODEL = None

def get_maintenance_context(query: str, maintenance_pipeline: MaintenancePipeline) -> str:
    query_lower = query.lower()
    context_parts = []
    if any(kw in query_lower for kw in ['risk', 'failure', 'alert']):
        high_risk = maintenance_pipeline.get_equipment_by_risk('HIGH')
        if high_risk:
            context_parts.append("High-Risk Equipment Data:")
            for eq in high_risk[:5]:
                context_parts.append(f"- ID: {eq['id']}, Type: {eq['type']}, Location: {eq['location']}, Failure Probability: {eq['failure_probability']:.1%}")
    elif any(kw in query_lower for kw in ['schedule', 'calendar', 'upcoming']):
        schedule = maintenance_pipeline.get_maintenance_schedule(30)
        if schedule:
            context_parts.append("Upcoming Maintenance Schedule (Next 30 Days):")
            for item in schedule[:8]:
                # FIX: Corrected item['date'] to item['next_maintenance']
                context_parts.append(f"- ID: {item['id']}, Date: {item['next_maintenance']}, Location: {item['location']}, Cost: ${item['maintenance_cost']}")
    return "\n".join(context_parts)

def generate_llm_response(query: str, search_results: List[Dict], maintenance_pipeline: MaintenancePipeline) -> str:
    if not GEMINI_MODEL:
        return "The AI model is not configured. Please check your API key."
    doc_context = ""
    if search_results:
        doc_context += "Relevant information from your documents:\n"
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
- The maintenance schedule or high-risk items.
- Information from your uploaded documents (e.g., "troubleshooting guide for cameras").
📊 **Current System Status:**
- 🚨 High risk equipment: {high_risk_count}
- 📅 Maintenance due this week: {upcoming_maintenance}"""

    prompt = f"""You are an expert AI assistant. Your goal is to provide clear, concise answers based ONLY on the context provided below.
**User's Question:** "{query}"
---
**Context from Documents:**
{doc_context if doc_context else "No relevant document context found."}
---
**Context from Predictive Maintenance System:**
{maintenance_context if maintenance_context else "No relevant maintenance data found for this specific query."}
---
Based on all the context above, provide a direct and helpful answer."""
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred while generating the AI response: {e}")
        return "Sorry, I encountered an issue while trying to answer your question."

# --- Document Processing and Search ---
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

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.warning(f"Could not read {os.path.basename(file_path)} with PyMuPDF. Error: {e}")
        return ""

def extract_text_from_email(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                try:
                    return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                except: continue
    elif msg.get_content_type() == "text/plain":
        try:
            return msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')
        except: return ""
    return ""

def format_email_for_rag(msg) -> str:
    body = extract_text_from_email(msg)
    if not body: return ""
    return f"""--- Email ---
From: {msg['From']}
To: {msg['To']}
Subject: {msg['Subject']}
Date: {msg['Date']}
{body.strip()}
--- End Email ---"""

@st.cache_resource
def load_and_process_documents():
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox"]
    all_files = [f for pattern in file_patterns for f in glob.glob(pattern, recursive=True)]
    docs, file_paths = [], []

    progress_bar = st.progress(0, text="Loading documents...")
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        progress_bar.progress((i + 1) / len(all_files), text=f"Processing: {file_name}")
        content = ""
        if file_path.endswith('.pdf'): content = extract_text_from_pdf(file_path)
        elif file_path.endswith('.eml'):
            with open(file_path, 'rb') as f: content = format_email_for_rag(BytesParser(policy=policy.default).parse(f))
        elif file_path.endswith('.mbox'):
            mbox_content = [format_email_for_rag(msg) for msg in mailbox.mbox(file_path)]
            content = "\n\n".join(filter(None, mbox_content))
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
            except:
                with open(file_path, 'r', encoding='latin-1') as f: content = f.read()
        
        if content and len(content.strip()) > 50:
            docs.append(clean_text(content))
            file_paths.append(file_path)
    progress_bar.empty()
    if docs: st.success(f"Successfully loaded and processed {len(docs)} documents!")
    return docs, file_paths

@st.cache_resource
def create_search_index(_documents, _file_paths):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    all_chunks, chunk_metadata = [], []
    for i, doc in enumerate(_documents):
        chunks = smart_chunking(doc, config.chunk_size)
        all_chunks.extend(chunks)
        chunk_metadata.extend([{'path': _file_paths[i]}] * len(chunks))
    if not all_chunks: return None, None, [], []
    with st.spinner("Embedding documents for semantic search..."):
        embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    st.success(f"✅ Search index created with {len(all_chunks)} text chunks.")
    return index, model, all_chunks, chunk_metadata

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    if not query or index is None: return []
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and scores[0][i] > config.similarity_threshold:
             results.append({'content': chunks[idx], 'source': metadata[idx]['path'], 'similarity': scores[0][i]})
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# --- Main App Interface ---
st.title("🤖 AI Support Chatbot")
st.markdown("Ask questions about your documents, emails, or maintenance schedules.")

try:
    maintenance_pipeline = MaintenancePipeline()
    docs, paths = load_and_process_documents()
    if docs:
        search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths)
    else:
        search_index, model, all_chunks, chunk_meta = None, None, [], []
        st.warning("No documents found. The chatbot will only use maintenance data.")
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    search_index, maintenance_pipeline = None, None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            search_results = search_documents(prompt, search_index, model, all_chunks, chunk_meta)
            response = generate_llm_response(prompt, search_results, maintenance_pipeline)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
