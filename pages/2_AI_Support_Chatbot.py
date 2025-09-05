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

# Optional: For dark mode toggle
try:
    from streamlit_extras.switch_page_button import switch_page
    from streamlit_extras.stoggle import stoggle
except ImportError:
    pass

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Settings ---
@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.4

if "settings" not in st.session_state:
    st.session_state.settings = {"chunk_size": 500, "theme": "light"}

config = RAGConfig(chunk_size=st.session_state.settings["chunk_size"])

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

# --- Tool 1: Data Analysis Tool ---
def analyze_csv_data(query: str) -> str:
    try:
        jira_file = next(f for f in glob.glob("*.csv") if "Closed tickets" in f)
    except StopIteration:
        return "The Jira ticket export CSV file was not found."
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
                if "rca" in col.lower() or "root cause" in col.lower():
                    target_column = col
                    break
            if not target_column:
                return "Could not determine the 'Root Cause Analysis' column in the CSV."
            count = df[df[target_column].str.contains(target_value, case=False, na=False)].shape[0]
            return f"Found **{count} tickets** related to **'{target_value}'**."
        return "I can count issues from the Jira file, but I didn't understand what to count. Try 'how many Device Faulty issues?'"
    except Exception as e:
        return f"An error occurred during CSV analysis: {e}"

# --- Tool 2: Maintenance and RAG Tool ---
def get_maintenance_context_with_actions(query: str, maintenance_pipeline: MaintenancePipeline, search_func, search_args) -> str:
    query_lower = query.lower()
    context_parts = []
    if any(kw in query_lower for kw in ['risk', 'failure', 'alert']):
        high_risk = maintenance_pipeline.get_equipment_by_risk('HIGH')
        if high_risk:
            context_parts.append("### High-Risk Equipment Alerts:\n")
            for eq in high_risk[:3]:
                context_parts.append(f"- **ID:** `{eq['id']}` | **Location:** {eq['location']} | **Failure Probability:** {eq['failure_probability']:.1%}")
                action_query = f"troubleshooting procedure for {eq['type']}"
                action_results = search_func(action_query, **search_args)
                if action_results:
                    action_text = action_results[0]['content'].strip().replace('\n', ' ')
                    source = os.path.basename(action_results[0]['source'])
                    context_parts.append(f"  - **▶️ Recommended Action:** Based on `{source}`, you should: *\"{action_text[:200]}...\"*")
                else:
                    context_parts.append("  - *No specific troubleshooting guide found in the documents.*")
    elif any(kw in query_lower for kw in ['schedule', 'calendar', 'upcoming']):
        schedule = maintenance_pipeline.get_maintenance_schedule(30)
        if schedule:
            context_parts.append("### Upcoming Maintenance (Next 30 Days):\n")
            for item in schedule[:5]:
                context_parts.append(f"- **{item['next_maintenance']}**: `{item['id']}` at {item['location']} (Cost: ${item['maintenance_cost']})")
    return "\n".join(context_parts)

# --- LLM and Response Generation (Upgraded) ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    GEMINI_MODEL = None

def summarize_history(history: List[Dict], max_tokens=300) -> str:
    # Simple summarizer: join last few messages, truncate if too long
    text = ""
    for msg in history[-6:]:
        text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return text[-max_tokens:]

def generate_response_stream(query: str, chat_history: List[Dict], context: str) -> Generator:
    if not GEMINI_MODEL:
        yield "The AI model is not configured. Please check your API key."
        return
    history_prompt = summarize_history(chat_history)
    prompt = f"""You are an expert AI support assistant. Your goal is to provide clear, concise answers.
Use the chat history for context but rely ONLY on the 'Context for your answer' to formulate your response. Cite sources if available.

---
**Chat History:**
{history_prompt if history_prompt else "This is the start of the conversation."}
---
**Context for your answer:**
{context if context else "No specific context was found for this query."}
---
**User's Question:** "{query}"
---
Based on all the information above, provide a direct and helpful answer. If the context is empty, explain what the user can ask about."""
    try:
        response_stream = GEMINI_MODEL.generate_content(prompt, stream=True)
        for chunk in response_stream:
            yield chunk.text
    except Exception as e:
        yield f"An error occurred while generating the response: {e}"

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
        st.warning(f"Could not read {os.path.basename(file_path)} with PyMuPDF: {e}")
        return ""

def extract_text_from_email(msg) -> str:
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
    body = extract_text_from_email(msg)
    if not body: return ""
    return f"--- Email ---\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\nDate: {msg['Date']}\n{body.strip()}\n--- End Email ---"

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

# --- Sidebar: Quick Actions, Upload, Settings ---
with st.sidebar:
    st.title("🤖 AI Support Assistant")
    st.markdown("**Quick Actions:**")
    if st.button("Show High-Risk Alerts"):
        st.session_state.messages.append({"role": "user", "content": "Show high-risk equipment alerts"})
    if st.button("Upcoming Maintenance"):
        st.session_state.messages.append({"role": "user", "content": "Show upcoming maintenance"})
    st.markdown("---")
    st.markdown("**System Status:**")
    maintenance_pipeline = MaintenancePipeline()
    st.write(f"🚨 {len(maintenance_pipeline.get_equipment_by_risk('HIGH'))} high risk items")
    st.write(f"📅 {len(maintenance_pipeline.get_maintenance_schedule(7))} tasks due this week")
    st.markdown("---")
    st.info("Upload new documents below:")
    uploaded_files = st.file_uploader("Add files", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded! Please reload the app to re-index documents.")
    st.markdown("---")
    st.markdown("**Settings:**")
    chunk_size = st.slider("Chunk size for document splitting", 200, 1000, st.session_state.settings["chunk_size"], 100)
    st.session_state.settings["chunk_size"] = chunk_size
    # Optional: Theme toggle
    if "theme" in st.session_state.settings:
        theme = st.radio("Theme", ["light", "dark"], index=0 if st.session_state.settings["theme"]=="light" else 1)
        st.session_state.settings["theme"] = theme

# --- Chat Bubble Styling ---
def chat_bubble(content, is_user=False):
    color = "#DCF8C6" if is_user else "#F1F0F0"
    align = "right" if is_user else "left"
    st.markdown(
        f"""
        <div style='background-color:{color};padding:10px;border-radius:10px;margin:5px 0;max-width:80%;float:{align};clear:both;'>
            {content}
        </div>
        """, unsafe_allow_html=True
    )

# --- Typing Animation for Streaming ---
def typing_animation():
    st.markdown("<i>Assistant is typing...</i>", unsafe_allow_html=True)

# --- Feedback Buttons ---
def feedback_buttons():
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("👍", key="thumbs_up"):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎", key="thumbs_down"):
            st.warning("We'll try to improve!")

# --- Intent Classification ---
def classify_intent(prompt):
    prompt = prompt.lower()
    if any(kw in prompt for kw in ["how many", "count", "total tickets"]):
        return "csv"
    if any(kw in prompt for kw in ['risk', 'failure', 'alert', 'schedule', 'calendar', 'upcoming']):
        return "maintenance"
    return "search"

# --- Initialization ---
try:
    docs, paths = load_and_process_documents()
    if docs:
        search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths)
        search_args = {'index': search_index, 'model': model, 'chunks': all_chunks, 'metadata': chunk_meta}
    else:
        search_index, model, all_chunks, chunk_meta, search_args = None, None, [], [], {}
        st.warning("No documents found. Document search and actionable insights are disabled.")
except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    search_index, search_args = None, {}

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]

# --- Chat History Display ---
for message in st.session_state.messages:
    chat_bubble(message["content"], is_user=(message["role"]=="user"))

# --- Chat Input and Tool Routing ---
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_bubble(prompt, is_user=True)
    with st.spinner("Thinking..."):
        intent = classify_intent(prompt)
        context = ""
        if intent == "csv":
            context = analyze_csv_data(prompt)
        elif intent == "maintenance":
            if search_args:
                context = get_maintenance_context_with_actions(prompt, maintenance_pipeline, search_documents, search_args)
            else:
                context = "Maintenance data is available, but I can't provide actionable insights without documents."
        else:
            if search_args:
                search_results = search_documents(prompt, **search_args)
                if search_results:
                    context += "### Relevant Information from Documents:\n"
                    for result in search_results[:3]:
                        source = os.path.basename(result['source'])
                        content_preview = result['content'].strip().replace('\n', ' ')
                        context += f"- **From `{source}`**: \"{content_preview}\"\n"
                    context += "\n_Citations provided above._"
        if not context:
            high_risk_count = len(maintenance_pipeline.get_equipment_by_risk('HIGH'))
            upcoming_maintenance = len(maintenance_pipeline.get_maintenance_schedule(7))
            context = f"""I couldn't find specific information for that query.
- You can ask about document contents (e.g., "summarize the INDIS meeting").
- You can ask for maintenance alerts or schedules.
- You can ask to count Jira tickets (e.g., "how many Device Faulty issues?").

**Current System Status:** 🚨 {high_risk_count} high risk items | 📅 {upcoming_maintenance} tasks due this week."""
        # --- Stream LLM Response with Typing Animation ---
        response_stream = generate_response_stream(prompt, st.session_state.messages, context)
        full_response = ""
        for chunk in response_stream:
            full_response += chunk
            chat_bubble(full_response)
            typing_animation()
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        feedback_buttons()
