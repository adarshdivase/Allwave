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
from typing import List, Dict, Any, Generator
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

# --- Basic Configuration ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Proactive AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced RAG Configuration ---
@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 3 # Reduced for more focused context
    similarity_threshold: float = 0.4

config = RAGConfig()

# --- Predictive Maintenance Integration (Simulation) ---
class MaintenancePipeline:
    def __init__(self):
        self.maintenance_data = self._load_maintenance_data()
        self.equipment_list = list(self.maintenance_data.keys())

    def _load_maintenance_data(self) -> Dict:
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'Network Switch', 'Server']
        for i in range(30): # Increased equipment count
            eq_type = random.choice(equipment_types)
            fail_prob = random.uniform(0.1, 0.9)
            equipment_data[f"{eq_type.replace(' ','_')}_{i+1}"] = {
                'type': eq_type,
                'location': f"Building A - Floor {random.randint(1,4)} - Rack {chr(65+i%4)}",
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

# --- Tool 1: Data Analysis Tool ---
def analyze_csv_data(query: str) -> str:
    # ... (function is unchanged from previous version) ...
    try:
        jira_file = next(f for f in glob.glob("*.csv") if "tickets" in f.lower())
    except StopIteration:
        return "No ticket CSV file was found."
    # ... (rest of the function is the same)
    try:
        df = pd.read_csv(jira_file)
        query_lower = query.lower()
        match = re.search(r"count of (.+?) issues|how many (.+?) issues", query_lower)
        if not match:
             match = re.search(r"related to a '(.+?)' issue", query_lower)
        if match:
            target_value = next(g for g in match.groups() if g is not None).strip()
            target_column = None
            for col in df.columns:
                if "rca" in col.lower() or "root cause" in col.lower() or "summary" in col.lower():
                    target_column = col
                    break
            if not target_column: return "Could not determine the 'Root Cause' or 'Summary' column in the CSV."
            count = df[df[target_column].str.contains(target_value, case=False, na=False)].shape[0]
            return f"Found **{count} tickets** related to **'{target_value}'** in `{os.path.basename(jira_file)}`."
        return "I can count issues from the Jira file, but I didn't understand what to count. Try 'how many Device Faulty issues?'"
    except Exception as e:
        return f"An error occurred during CSV analysis: {e}"


# --- LLM and Response Generation (Upgraded for Vision & Streaming) ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    GEMINI_MODEL = None

def generate_response_stream(prompt_parts: List[Any]) -> Generator:
    if not GEMINI_MODEL:
        yield "The AI model is not configured. Please check your API key."
        return
    try:
        response_stream = GEMINI_MODEL.generate_content(prompt_parts, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"An error occurred while generating the response: {e}"

# --- Document Processing and Search (Unchanged) ---
# ... (All functions from clean_text to search_documents remain the same as the previous version) ...
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip())

def smart_chunking(text: str, chunk_size: int) -> List[str]:
    # ... (implementation unchanged)
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
    # ... (implementation unchanged)
    try:
        with fitz.open(file_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.warning(f"Could not read {os.path.basename(file_path)} with PyMuPDF: {e}")
        return ""

def extract_text_from_email(msg) -> str:
    # ... (implementation unchanged)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
                try: return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')
                except: continue
    elif msg.get_content_type() == "text/plain":
        try: return msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')
        except: return ""
    return ""

def format_email_for_rag(msg) -> str:
    # ... (implementation unchanged)
    body = extract_text_from_email(msg)
    if not body: return ""
    return f"--- Email ---\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\nDate: {msg['Date']}\n{body.strip()}\n--- End Email ---"

@st.cache_resource
def load_and_process_documents():
    # ... (implementation unchanged)
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox", "**/*.docx", "**/*.pptx"]
    all_files = [f for pattern in file_patterns for f in glob.glob(pattern, recursive=True)]
    docs, file_paths = [], []
    progress_bar = st.progress(0, text="Loading knowledge base...")
    # Add parsers for docx and pptx if you have them, otherwise they will be skipped
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        progress_bar.progress((i + 1) / len(all_files), text=f"Processing: {file_name}")
        content = ""
        try:
            if file_path.endswith('.pdf'): content = extract_text_from_pdf(file_path)
            elif file_path.endswith('.eml'):
                with open(file_path, 'rb') as f: content = format_email_for_rag(BytesParser(policy=policy.default).parse(f))
            elif file_path.endswith('.mbox'):
                mbox_content = [format_email_for_rag(msg) for msg in mailbox.mbox(file_path)]
                content = "\n\n".join(filter(None, mbox_content))
            elif file_path.endswith(('.txt', '.md', '.csv')):
                with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        except Exception as e:
            st.warning(f"Skipping {file_name}: {e}")
            continue
        if content and len(content.strip()) > 50:
            docs.append(clean_text(content))
            file_paths.append(file_path)
    progress_bar.empty()
    if docs: st.success(f"Knowledge base loaded with {len(docs)} documents.")
    return docs, file_paths

@st.cache_resource
def create_search_index(_documents, _file_paths):
    # ... (implementation unchanged)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    all_chunks, chunk_metadata = [], []
    for i, doc in enumerate(_documents):
        chunks = smart_chunking(doc, config.chunk_size)
        all_chunks.extend(chunks)
        chunk_metadata.extend([{'path': _file_paths[i]}] * len(chunks))
    if not all_chunks: return None, None, [], []
    with st.spinner("Embedding documents..."):
        embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    st.success(f"✅ Search index created with {len(all_chunks)} text chunks.")
    return index, model, all_chunks, chunk_metadata

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    # ... (implementation unchanged)
    if not query or index is None: return []
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and scores[0][i] > config.similarity_threshold:
             results.append({'content': chunks[idx], 'source': metadata[idx]['path'], 'similarity': scores[0][i]})
    return sorted(results, key=lambda x: x['similarity'], reverse=True)


# --- NEW: Real-Time Alert Simulation & Diagnostic Agent ---
def simulate_real_time_alert(pipeline: MaintenancePipeline) -> Dict:
    """Generates a plausible, random, real-time alert."""
    alert_type = random.choice(["Device Offline", "High Temperature", "High CPU Usage"])
    device = random.choice(pipeline.equipment_list)
    device_info = pipeline.maintenance_data[device]
    return {
        "alert_type": alert_type,
        "device_id": device,
        "device_type": device_info['type'],
        "location": device_info['location'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def run_diagnostic_agent(alert: Dict, search_func, search_args, pipeline: MaintenancePipeline) -> str:
    """The 'Cognitive Loop' that gathers context about an alert."""
    st.info(f"**Agent Activated!** Diagnosing alert for `{alert['device_id']}`...")
    
    context_parts = [f"**Real-Time Alert Details:**\n- **Time**: {alert['timestamp']}\n- **Device**: `{alert['device_id']}` ({alert['device_type']})\n- **Location**: {alert['location']}\n- **Issue**: {alert['alert_type']}\n"]

    # 1. Search for Standard Operating Procedures (SOPs)
    sop_query = f"troubleshooting procedure for {alert['device_type']} {alert['alert_type']}"
    sop_results = search_func(sop_query, **search_args)
    if sop_results:
        context_parts.append(f"**Relevant SOP Found:**\n- From `{os.path.basename(sop_results[0]['source'])}`: *'{sop_results[0]['content']}'*")

    # 2. Check Maintenance History
    device_info = pipeline.maintenance_data[alert['device_id']]
    context_parts.append(f"**Maintenance System Data:**\n- **Next Scheduled Maintenance**: {device_info['next_maintenance']}\n- **Current Failure Probability**: {device_info['failure_probability']:.1%}")

    # 3. Search for Similar Past Issues in tickets
    ticket_query = f"{alert['device_type']} {alert['alert_type']}"
    ticket_results = search_func(ticket_query, **search_args)
    if any("tickets" in res['source'].lower() for res in ticket_results):
        ticket_context = next(res for res in ticket_results if "tickets" in res['source'].lower())
        context_parts.append(f"**Similar Past Ticket Found:**\n- From `{os.path.basename(ticket_context['source'])}`: *'{ticket_context['content']}'*")

    return "\n\n".join(context_parts)

# --- Main App Interface (Upgraded) ---
st.title("🤖 Proactive AI Support Assistant")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    st.write("Upload images or trigger a simulated alert to see the AI agent in action.")
    
    # NEW: Real-Time Alert Simulation Button
    if st.button("🚨 Simulate Real-Time Alert", use_container_width=True):
        st.session_state.simulated_alert = simulate_real_time_alert(st.session_state.maintenance_pipeline)
    
    st.markdown("---")
    st.caption("This app demonstrates advanced AI capabilities including RAG, data analysis, and multi-modal reasoning.")


# --- Initialization ---
# Use session state to store expensive-to-load objects
if "init_done" not in st.session_state:
    try:
        st.session_state.maintenance_pipeline = MaintenancePipeline()
        docs, paths = load_and_process_documents()
        if docs:
            search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths)
            st.session_state.search_args = {'index': search_index, 'model': model, 'chunks': all_chunks, 'metadata': chunk_meta}
        else:
            st.session_state.search_args = None
            st.warning("No documents found. Document search and diagnostics are disabled.")
        st.session_state.init_done = True
    except Exception as e:
        st.error(f"An error occurred during initialization: {e}")
        st.session_state.search_args = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you? You can ask me questions, upload an image, or trigger a real-time alert from the sidebar."}]

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Interaction Logic ---
# Check if an alert was triggered
if "simulated_alert" in st.session_state and st.session_state.simulated_alert:
    alert = st.session_state.simulated_alert
    st.session_state.simulated_alert = None # Consume the alert
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is diagnosing the alert..."):
            if st.session_state.search_args:
                context = run_diagnostic_agent(alert, search_documents, st.session_state.search_args, st.session_state.maintenance_pipeline)
                prompt_parts = [
                    "You are an AI diagnostic agent. You have received a real-time alert and gathered context from multiple sources. Your task is to synthesize this information into a concise, actionable report for a human operator. Provide a summary of the problem, a likely root cause, and a clear, numbered list of recommended actions. Be direct and professional.",
                    "---CONTEXT---",
                    context
                ]
                response_stream = generate_response_stream(prompt_parts)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error("Cannot run diagnostic agent because the search index is not available.")

# Handle chat input
prompt = st.chat_input("Ask a question or describe the image you're uploading...")
uploaded_image = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if prompt or uploaded_image:
    # Handle image upload
    image = None
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.session_state.messages.append({"role": "user", "content": f"Image uploaded: {uploaded_image.name}"})
        with st.chat_message("user"):
            st.image(image, width=200)
            if prompt: st.markdown(prompt) # Display prompt alongside image
    
    # Handle text prompt (if any)
    if prompt and not uploaded_image:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    # Start generating response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = ""
            prompt_lower = prompt.lower() if prompt else ""

            # --- Router: Decide which tool to use or if it's a general query ---
            if any(kw in prompt_lower for kw in ["how many", "count", "total tickets"]):
                context = analyze_csv_data(prompt)
                prompt_parts = [
                    "You are an AI assistant. The user asked a question that was answered by the data analysis tool. Present the result clearly.",
                    f"---TOOL RESULT---\n{context}",
                    f"\nUser's Original Question: {prompt}"
                ]
            else: # General RAG or Vision query
                if st.session_state.search_args:
                    search_results = search_documents(prompt, **st.session_state.search_args)
                    if search_results:
                        context += "### Relevant Information from Documents:\n"
                        for result in search_results:
                            source = os.path.basename(result['source'])
                            content_preview = result['content'].strip().replace('\n', ' ')
                            context += f"- **From `{source}`**: \"{content_preview}\"\n"
                
                # Construct the final prompt for the LLM
                prompt_parts = [
                     "You are an expert AI support assistant. Answer the user's question based on the provided image (if any) and the text context from relevant documents. If no context is found, say so.",
                     f"\n---DOCUMENT CONTEXT---\n{context if context else 'No relevant documents found.'}",
                ]
                if prompt: prompt_parts.append(f"\n---USER'S QUESTION---\n{prompt}")
                if image: prompt_parts.append(image)

            # Generate and Stream Response
            response_stream = generate_response_stream(prompt_parts)
            full_response = st.write_stream(response_stream)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
