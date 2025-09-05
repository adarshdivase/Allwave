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

# --- Intent Classification ---
def classify_intent(prompt):
    prompt = prompt.lower()
    if any(kw in prompt for kw in ["how many", "count", "total tickets"]):
        return "csv"
    if any(kw in prompt for kw in ['risk', 'failure', 'alert', 'schedule', 'calendar', 'upcoming']):
        return "maintenance"
    return "search"

# --- Initialization ---
# Moved up to ensure these are available for the sidebar
try:
    maintenance_pipeline = MaintenancePipeline()
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
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}]


# --- Sidebar: Quick Actions, Upload, Settings ---
with st.sidebar:
    st.title("🤖 AI Support Assistant")
    st.markdown("**Quick Actions:**")

    # Handler function for quick actions
    def handle_quick_action(action_type: str):
        if action_type == "alerts":
            prompt = "Show high-risk equipment alerts"
        else:
            prompt = "Show upcoming maintenance"
            
        now = datetime.now().strftime("%H:%M")
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": now
        })
        
        # Generate AI response context
        if action_type == "alerts":
            context = get_maintenance_context_with_actions(
                prompt, 
                maintenance_pipeline,
                search_documents,
                search_args
            )
        else:
            context = get_maintenance_context_with_actions(
                prompt,
                maintenance_pipeline,
                search_documents,
                search_args
            )
            
        # Generate and add AI response
        response_stream = generate_response_stream(prompt, st.session_state.messages, context)
        full_response = "".join([chunk for chunk in response_stream])
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": now
        })
        # Force a rerun to show the new messages
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Risk Alerts", key="risk_btn", help="Show equipment with the highest failure probability."):
            handle_quick_action("alerts")
    with col2:
        if st.button("📅 Maintenance", key="maint_btn", help="Show the upcoming maintenance schedule."):
            handle_quick_action("maintenance")

    # Rest of your sidebar code
    st.markdown("---")
    st.markdown("**System Status:**")
    st.write(f"🚨 {len(maintenance_pipeline.get_equipment_by_risk('HIGH'))} high risk items")
    st.write(f"📅 {len(maintenance_pipeline.get_maintenance_schedule(7))} tasks due this week")
    st.markdown("---")
    st.info("Upload new documents below:")
    uploaded_files = st.file_uploader("Add files", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save to a directory for persistent storage if needed
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded! Please reload the app to re-index documents.")
        st.button("Reload App")
    st.markdown("---")
    st.markdown("**Settings:**")
    chunk_size = st.slider("Chunk size for document splitting", 200, 1000, st.session_state.settings["chunk_size"], 100)
    st.session_state.settings["chunk_size"] = chunk_size
    theme = st.radio("Theme", ["light", "dark"], index=0 if st.session_state.settings["theme"]=="light" else 1)
    st.session_state.settings["theme"] = theme

# --- Improved Chat Bubble with Avatar and Timestamp ---
def chat_bubble(content, is_user=False, timestamp=None):
    theme = st.session_state.settings.get("theme", "light")
    if theme == "dark":
        user_color = "#3A6351"
        assistant_color = "#222831"
        text_color = "#fff"
        border = "1px solid #393E46"
    else:
        user_color = "#DCF8C6"
        assistant_color = "#F1F0F0"
        text_color = "#222"
        border = "1px solid #ddd"
    
    color = user_color if is_user else assistant_color
    align = "right" if is_user else "left"
    avatar_url = (
        "https://cdn-icons-png.flaticon.com/512/1946/1946429.png" if is_user
        else "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
    )
    name = "You" if is_user else "AI"
    if not timestamp:
        timestamp = datetime.now().strftime("%H:%M")
    
    # Sanitize content for HTML display
    content = content.replace("<", "&lt;").replace(">", "&gt;")

    st.markdown(
        f"""
        <div style='display:flex;flex-direction:{"row-reverse" if is_user else "row"};align-items:flex-end;margin:8px 0;'>
            <img src="{avatar_url}" width="36" height="36" style="border-radius:50%;margin:0 8px;">
            <div style='background-color:{color};
                        color:{text_color};
                        padding:14px 16px;
                        border-radius:12px;
                        max-width:70vw;
                        min-width:80px;
                        border:{border};
                        box-shadow:0 2px 8px rgba(0,0,0,0.07);
                        position:relative;'>
                <div style='font-size:13px;font-weight:bold;opacity:0.7;'>{name}</div>
                <div style='font-size:16px;margin:4px 0;'>{content}</div>
                <div style='font-size:11px;opacity:0.5;text-align:right;'>{timestamp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# --- Feedback Buttons ---
def feedback_buttons():
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("👍", key="thumbs_up"):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎", key="thumbs_down"):
            st.warning("We'll try to improve!")


# --- Chat History Display with Avatars and Timestamps ---
for message in st.session_state.messages:
    if "timestamp" not in message:
        message["timestamp"] = datetime.now().strftime("%H:%M")
    chat_bubble(
        message["content"],
        is_user=(message["role"] == "user"),
        timestamp=message["timestamp"]
    )

# --- Chat Input and Tool Routing ---
if prompt := st.chat_input("Ask a question..."):
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": now})
    # Display the user's message immediately
    chat_bubble(prompt, is_user=True, timestamp=now)
    
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
        else: # "search" intent
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

        # Generate and display the AI's response
        with st.empty():
            response_stream = generate_response_stream(prompt, st.session_state.messages, context)
            full_response = ""
            for chunk in response_stream:
                full_response += chunk
                chat_bubble(full_response + " ▌", is_user=False, timestamp=now)
            
            # Final update to remove the cursor
            chat_bubble(full_response, is_user=False, timestamp=now)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": now})
        feedback_buttons()
        st.rerun()



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

import fitz  # PyMuPDF

from datetime import datetime, timedelta

import random

from dataclasses import dataclass

import google.generativeai as genai

import mailbox

from email import policy

from email.parser import BytesParser



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

        # Set random seed for consistent demo data

        random.seed(42)

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

        jira_files = [f for f in glob.glob("*.csv") if "Closed tickets" in f or "jira" in f.lower() or "tickets" in f.lower()]

        if not jira_files:

            return "No Jira ticket CSV files found. Please ensure the CSV file contains 'Closed tickets' in the filename or upload a tickets CSV file."

        

        jira_file = jira_files[0]

        df = pd.read_csv(jira_file)

        

        query_lower = query.lower()

        

        # Enhanced pattern matching for various query formats

        patterns = [

            r"count of (.+?) issues",

            r"how many (.+?) issues",

            r"(.+?) issue count",

            r"related to a?n? '(.+?)' issue",

            r"(.+?) tickets"

        ]

        

        target_value = None

        for pattern in patterns:

            match = re.search(pattern, query_lower)

            if match:

                target_value = next(g for g in match.groups() if g is not None).strip()

                break

        

        if not target_value:

            return f"I found the CSV file '{os.path.basename(jira_file)}' but couldn't understand what to count. Try asking 'How many Device Faulty issues?' or 'Count of Network issues'."

        

        # Find the appropriate column for analysis

        target_column = None

        possible_columns = ['root cause analysis', 'rca', 'issue type', 'category', 'problem type', 'cause']

        for col in df.columns:

            if any(pc in col.lower() for pc in possible_columns):

                target_column = col

                break

        

        if not target_column:

            available_cols = ", ".join(df.columns[:5])

            return f"Could not find a suitable analysis column in the CSV. Available columns include: {available_cols}..."

        

        # Perform the search and count

        matching_rows = df[df[target_column].str.contains(target_value, case=False, na=False)]

        count = matching_rows.shape[0]

        

        if count > 0:

            return f"Found **{count} tickets** related to **'{target_value}'** in the '{target_column}' column."

        else:

            return f"No tickets found related to **'{target_value}'** in the '{target_column}' column. The file contains {len(df)} total tickets."

            

    except Exception as e:

        return f"An error occurred during CSV analysis: {str(e)}. Please ensure the CSV file is properly formatted."



# --- Tool 2: Maintenance and RAG Tool ---

def get_maintenance_context_with_actions(query: str, maintenance_pipeline: MaintenancePipeline, search_func=None, search_args=None) -> str:

    query_lower = query.lower()

    context_parts = []

    

    if any(kw in query_lower for kw in ['risk', 'failure', 'alert']):

        high_risk = maintenance_pipeline.get_equipment_by_risk('HIGH')

        if high_risk:

            context_parts.append("### High-Risk Equipment Alerts:\n")

            for eq in high_risk[:3]:

                context_parts.append(f"- **ID:** `{eq['id']}` | **Location:** {eq['location']} | **Failure Probability:** {eq['failure_probability']:.1%}")

                

                # Only attempt to search for actions if search functionality is available

                if search_func and search_args:

                    action_query = f"troubleshooting procedure for {eq['type']}"

                    action_results = search_func(action_query, **search_args)

                    if action_results:

                        action_text = action_results[0]['content'].strip().replace('\n', ' ')

                        source = os.path.basename(action_results[0]['source'])

                        context_parts.append(f"  - **▶️ Recommended Action:** Based on `{source}`: *\"{action_text[:200]}...\"*")

                    else:

                        context_parts.append("  - *No specific troubleshooting guide found in documents.*")

                else:

                    context_parts.append("  - *Document search not available for specific procedures.*")

                    

    elif any(kw in query_lower for kw in ['schedule', 'calendar', 'upcoming']):

        schedule = maintenance_pipeline.get_maintenance_schedule(30)

        if schedule:

            context_parts.append("### Upcoming Maintenance (Next 30 Days):\n")

            for item in schedule[:5]:

                context_parts.append(f"- **{item['next_maintenance']}**: `{item['id']}` at {item['location']} (Cost: ${item['maintenance_cost']})")

        else:

            context_parts.append("### No maintenance scheduled for the next 30 days.")

    

    return "\n".join(context_parts)



# --- LLM and Response Generation (Upgraded) ---

try:

    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')

except Exception as e:

    st.error(f"Error configuring Gemini API: {e}")

    GEMINI_MODEL = None



def summarize_history(history: List[Dict], max_tokens=300) -> str:

    text = ""

    for msg in history[-6:]:

        text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    return text[-max_tokens:]



def generate_response_stream(query: str, chat_history: List[Dict], context: str) -> Generator:

    if not GEMINI_MODEL:

        yield "The AI model is not configured. Please check your API key."

        return

    

    history_prompt = summarize_history(chat_history)

    # This prompt encourages markdown for better readability, taken from the previous version.

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

            if chunk.text:  # Ensure chunk has text content

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

            if current_chunk: 

                chunks.append(current_chunk.strip())

            current_chunk = sentence + " "

    if current_chunk: 

        chunks.append(current_chunk.strip())

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

                try: 

                    return part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8')

                except: 

                    continue

    elif msg.get_content_type() == "text/plain":

        try: 

            return msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8')

        except: 

            return ""

    return ""



def format_email_for_rag(msg) -> str:

    body = extract_text_from_email(msg)

    if not body: 

        return ""

    return f"--- Email ---\nFrom: {msg['From']}\nTo: {msg['To']}\nSubject: {msg['Subject']}\nDate: {msg['Date']}\n{body.strip()}\n--- End Email ---"



@st.cache_resource

def load_and_process_documents():

    # Look in root and in 'uploads' directory

    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox"]

    all_files = [f for pattern in file_patterns for f in glob.glob(pattern, recursive=True)]

    docs, file_paths = [], []

    

    if not all_files:

        st.info("No documents found. Upload some files to enable document search.")

        return docs, file_paths

    

    progress_bar = st.progress(0, text="Loading documents...")

    

    for i, file_path in enumerate(all_files):

        file_name = os.path.basename(file_path)

        progress_bar.progress((i + 1) / len(all_files), text=f"Processing: {file_name}")

        content = ""

        

        try:

            if file_path.endswith('.pdf'): 

                content = extract_text_from_pdf(file_path)

            elif file_path.endswith('.eml'):

                with open(file_path, 'rb') as f: 

                    content = format_email_for_rag(BytesParser(policy=policy.default).parse(f))

            elif file_path.endswith('.mbox'):

                mbox_content = [format_email_for_rag(msg) for msg in mailbox.mbox(file_path)]

                content = "\n\n".join(filter(None, mbox_content))

            else:

                try:

                    with open(file_path, 'r', encoding='utf-8') as f: 

                        content = f.read()

                except:

                    with open(file_path, 'r', encoding='latin-1') as f: 

                        content = f.read()

                        

            if content and len(content.strip()) > 50:

                docs.append(clean_text(content))

                file_paths.append(file_path)

        except Exception as e:

            st.warning(f"Could not process {file_name}: {e}")

            continue

    

    progress_bar.empty()

    if docs: 

        st.success(f"Successfully loaded and processed {len(docs)} documents!")

    return docs, file_paths



@st.cache_resource

def create_search_index(_documents, _file_paths):

    if not _documents:

        return None, None, [], []

        

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    all_chunks, chunk_metadata = [], []

    

    for i, doc in enumerate(_documents):

        chunks = smart_chunking(doc, config.chunk_size)

        all_chunks.extend(chunks)

        chunk_metadata.extend([{'path': _file_paths[i]}] * len(chunks))

    

    if not all_chunks: 

        return None, None, [], []

    

    with st.spinner("Embedding documents for semantic search..."):

        embeddings = model.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)

    

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings.astype('float32'))

    st.success(f"✅ Search index created with {len(all_chunks)} text chunks.")

    return index, model, all_chunks, chunk_metadata



def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:

    if not query or index is None: 

        return []

    

    query_embedding = model.encode([query], normalize_embeddings=True)

    scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)

    results = []

    

    for i, idx in enumerate(indices[0]):

        if idx != -1 and scores[0][i] > config.similarity_threshold:

             results.append({

                 'content': chunks[idx], 

                 'source': metadata[idx]['path'], 

                 'similarity': scores[0][i]

             })

    

    return sorted(results, key=lambda x: x['similarity'], reverse=True)



# --- Intent Classification ---

def classify_intent(prompt):

    prompt = prompt.lower()

    if any(kw in prompt for kw in ["how many", "count", "total tickets", "tickets", "jira"]):

        return "csv"

    if any(kw in prompt for kw in ['risk', 'failure', 'alert', 'schedule', 'calendar', 'upcoming', 'maintenance']):

        return "maintenance"

    return "search"



# --- Initialization ---

try:

    maintenance_pipeline = MaintenancePipeline()

    docs, paths = load_and_process_documents()

    if docs:

        search_index, model, all_chunks, chunk_meta = create_search_index(docs, paths)

        search_args = {'index': search_index, 'model': model, 'chunks': all_chunks, 'metadata': chunk_meta}

    else:

        search_index, model, all_chunks, chunk_meta, search_args = None, None, [], [], {}

        if not docs:

            st.info("📁 No documents found. Document search will be limited until you upload files.")

except Exception as e:

    st.error(f"An error occurred during initialization: {e}")

    search_index, search_args = None, {}



if "messages" not in st.session_state:

    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?", "timestamp": datetime.now().strftime("%H:%M")}]





# --- Sidebar: Quick Actions, Upload, Settings ---

with st.sidebar:

    st.title("🤖 AI Support Assistant")

    st.markdown("**Quick Actions:**")



    # Handler function for quick actions

    def handle_quick_action(action_type: str):

        if action_type == "alerts":

            prompt = "Show high-risk equipment alerts"

        else:

            prompt = "Show upcoming maintenance schedule"

            

        now = datetime.now().strftime("%H:%M")

        # Add user message

        st.session_state.messages.append({

            "role": "user",

            "content": prompt,

            "timestamp": now

        })

        

        # Generate AI response context

        context = get_maintenance_context_with_actions(

            prompt, 

            maintenance_pipeline,

            search_documents if search_args else None,

            search_args if search_args else None

        )

            

        # Generate and add AI response

        response_stream = generate_response_stream(prompt, st.session_state.messages, context)

        full_response = "".join([chunk for chunk in response_stream])

        st.session_state.messages.append({

            "role": "assistant",

            "content": full_response,

            "timestamp": now

        })

        # Force a rerun to show the new messages

        st.rerun()



    col1, col2 = st.columns(2)

    with col1:

        if st.button("📊 Risk Alerts", key="risk_btn", help="Show equipment with the highest failure probability."):

            handle_quick_action("alerts")

    with col2:

        if st.button("📅 Maintenance", key="maint_btn", help="Show the upcoming maintenance schedule."):

            handle_quick_action("maintenance")



    # System status

    st.markdown("---")

    st.markdown("**System Status:**")

    high_risk_count = len(maintenance_pipeline.get_equipment_by_risk('HIGH'))

    upcoming_count = len(maintenance_pipeline.get_maintenance_schedule(7))

    st.write(f"🚨 {high_risk_count} high risk items")

    st.write(f"📅 {upcoming_count} tasks due this week")

    

    # File upload

    st.markdown("---")

    st.info("Upload new documents below:")

    uploaded_files = st.file_uploader("Add files", accept_multiple_files=True, 

                                      type=['txt', 'md', 'csv', 'pdf', 'eml', 'mbox'])

    if uploaded_files:

        upload_dir = "uploads"

        os.makedirs(upload_dir, exist_ok=True)

        for uploaded_file in uploaded_files:

            file_path = os.path.join(upload_dir, uploaded_file.name)

            with open(file_path, "wb") as f:

                f.write(uploaded_file.getbuffer())

        st.success(f"✅ {len(uploaded_files)} files uploaded!")

        if st.button("🔄 Reload App", help="Reload to re-index uploaded documents"):

            st.rerun()

    

    # Settings

    st.markdown("---")

    st.markdown("**Settings:**")

    chunk_size = st.slider("Chunk size for document splitting", 200, 1000, 

                           st.session_state.settings["chunk_size"], 100)

    st.session_state.settings["chunk_size"] = chunk_size

    

    theme = st.radio("Theme", ["light", "dark"], 

                     index=0 if st.session_state.settings["theme"]=="light" else 1)

    st.session_state.settings["theme"] = theme



# --- Improved Chat Bubble with timestamps ---

def chat_bubble(content, is_user=False, timestamp=None):

    theme = st.session_state.settings.get("theme", "light")

    if theme == "dark":

        user_color = "#3A6351"

        assistant_color = "#222831"

        text_color = "#fff"

        border = "1px solid #393E46"

    else:

        user_color = "#DCF8C6"

        assistant_color = "#F1F0F0"

        text_color = "#222"

        border = "1px solid #ddd"

    

    color = user_color if is_user else assistant_color

    avatar_url = (

        "https://cdn-icons-png.flaticon.com/512/1946/1946429.png" if is_user

        else "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"

    )

    name = "You" if is_user else "AI Assistant"

    if not timestamp:

        timestamp = datetime.now().strftime("%H:%M")

    

    # Sanitize content for HTML display

    content = content.replace("<", "&lt;").replace(">", "&gt;")



    st.markdown(

        f"""

        <div style='display:flex;flex-direction:{"row-reverse" if is_user else "row"};align-items:flex-end;margin:12px 0;'>

            <img src="{avatar_url}" width="36" height="36" style="border-radius:50%;margin:0 8px;flex-shrink:0;">

            <div style='background-color:{color};

                        color:{text_color};

                        padding:14px 16px;

                        border-radius:12px;

                        max-width:70%;

                        min-width:80px;

                        border:{border};

                        box-shadow:0 2px 8px rgba(0,0,0,0.07);

                        position:relative;'>

                <div style='font-size:13px;font-weight:bold;opacity:0.7;margin-bottom:6px;'>{name}</div>

                <div style='font-size:16px;line-height:1.4;'>{content}</div>

                <div style='font-size:11px;opacity:0.5;text-align:right;margin-top:8px;'>{timestamp}</div>

            </div>

        </div>

        """, unsafe_allow_html=True

    )



# --- Chat History Display ---

for message in st.session_state.messages:

    chat_bubble(

        message["content"],

        is_user=(message["role"] == "user"),

        timestamp=message.get("timestamp")

    )



# --- Chat Input and Tool Routing ---

if prompt := st.chat_input("Ask a question..."):

    now = datetime.now().strftime("%H:%M")

    # Add user message and display immediately

    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": now})

    chat_bubble(prompt, is_user=True, timestamp=now)

    

    with st.spinner("Thinking..."):

        intent = classify_intent(prompt)

        context = ""

        

        if intent == "csv":

            context = analyze_csv_data(prompt)

        elif intent == "maintenance":

            context = get_maintenance_context_with_actions(

                prompt, 

                maintenance_pipeline, 

                search_documents if search_args else None, 

                search_args if search_args else None

            )

        else:  # "search" intent

            if search_args:

                search_results = search_documents(prompt, **search_args)

                if search_results:

                    context += "### Relevant Information from Documents:\n"

                    for result in search_results[:3]:

                        source = os.path.basename(result['source'])

                        content_preview = result['content'].strip().replace('\n', ' ')

                        context += f"- **From `{source}`**: \"{content_preview[:300]}...\"\n"

                    context += "\n*Citations provided above.*"

                else:

                    context = "No relevant documents found for your query."

            else:

                context = "Document search is not available. Please upload documents to enable this feature."

        

        # Generate fallback context if nothing found

        if not context.strip():

            high_risk_count = len(maintenance_pipeline.get_equipment_by_risk('HIGH'))

            upcoming_maintenance = len(maintenance_pipeline.get_maintenance_schedule(7))

            context = f"""I couldn't find specific information for that query.



**What you can ask about:**

- Document contents (upload files first)

- Maintenance alerts and schedules

- Ticket counts from CSV files (e.g., "How many Device Faulty issues?")



**Current System Status:** 🚨 {high_risk_count} high risk equipment items  

📅 {upcoming_maintenance} maintenance tasks due this week"""



        # Check if we should render interactive content instead of regular AI response

        # NOTE: This is a placeholder for future interactive UI elements

        if context and ("INTERACTIVE_ALERTS" in context or "INTERACTIVE_SCHEDULE" in context):

            # Render interactive content directly

            # render_interactive_content(context) # This function would be defined elsewhere

            

            # Add a simple message to chat history

            if "INTERACTIVE_ALERTS" in context:

                summary_msg = "Displayed interactive high-risk equipment alerts with action buttons."

            else:

                summary_msg = "Displayed interactive maintenance schedule with management options."

                

            st.session_state.messages.append({"role": "assistant", "content": summary_msg, "timestamp": now})

        else:

            # Regular AI response generation

            response_placeholder = st.empty()

            full_response = ""

            

            try:

                response_stream = generate_response_stream(prompt, st.session_state.messages, context)

                for chunk in response_stream:

                    full_response += chunk

                    with response_placeholder.container():

                        chat_bubble(full_response + " ▌", is_user=False, timestamp=now)

                

                # Final update to remove cursor

                with response_placeholder.container():

                    chat_bubble(full_response, is_user=False, timestamp=now)

                    

            except Exception as e:

                full_response = f"I apologize, but I encountered an error: {str(e)}"

                with response_placeholder.container():

                    chat_bubble(full_response, is_user=False, timestamp=now)

            

            # Add to chat history

            st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": now})



# --- Footer ---

st.markdown("---")

st.markdown("*AI Support Assistant - Powered by Streamlit & Gemini*")

