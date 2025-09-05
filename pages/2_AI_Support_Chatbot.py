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
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import base64
import io

warnings.filterwarnings("ignore")

# Enhanced page configuration
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme variables */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #FFC107;
        --success-color: #4CAF50;
        --danger-color: #F44336;
        --warning-color: #FF9800;
        --info-color: #2196F3;
        --dark-bg: #1a1a1a;
        --light-bg: #ffffff;
        --border-radius: 12px;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --hover-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Enhanced card styling */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        border-left: 4px solid var(--primary-color);
        margin: 10px 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--hover-shadow);
    }
    
    .risk-high { border-left-color: var(--danger-color); }
    .risk-medium { border-left-color: var(--warning-color); }
    .risk-low { border-left-color: var(--success-color); }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-high { background-color: var(--danger-color); }
    .status-medium { background-color: var(--warning-color); }
    .status-low { background-color: var(--success-color); }
    
    /* Enhanced buttons */
    .stButton > button {
        border-radius: var(--border-radius);
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: var(--shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--hover-shadow);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        animation: slideIn 0.3s ease;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-right: 20%;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
    }
    
    /* Alert boxes */
    .alert {
        padding: 15px;
        border-radius: var(--border-radius);
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    .alert-info { 
        background: #e3f2fd; 
        border-left-color: var(--info-color);
        color: #0d47a1;
    }
    
    .alert-success { 
        background: #e8f5e8; 
        border-left-color: var(--success-color);
        color: #2e7d32;
    }
    
    .alert-warning { 
        background: #fff3e0; 
        border-left-color: var(--warning-color);
        color: #f57c00;
    }
    
    .alert-error { 
        background: #ffebee; 
        border-left-color: var(--danger-color);
        color: #c62828;
    }
    
    /* Loading animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced tables */
    .dataframe {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Load custom CSS
load_custom_css()

# --- Settings ---
@dataclass
class RAGConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.4

if "settings" not in st.session_state:
    st.session_state.settings = {
        "chunk_size": 500, 
        "theme": "light",
        "auto_scroll": True,
        "show_sources": True,
        "animation_enabled": True
    }

config = RAGConfig(chunk_size=st.session_state.settings["chunk_size"])

# --- Enhanced Predictive Maintenance Integration ---
class MaintenancePipeline:
    def __init__(self):
        self.maintenance_data = self._load_maintenance_data()

    def _load_maintenance_data(self) -> Dict:
        random.seed(42)
        equipment_data = {}
        equipment_types = ['HVAC', 'IT_EQUIPMENT', 'ELECTRICAL', 'FIRE_SAFETY', 'SECURITY', 'PLUMBING']
        locations = ['Building A - Floor 1', 'Building A - Floor 2', 'Building B - Floor 1', 'Building C - Basement']
        
        for i in range(25):
            eq_type = random.choice(equipment_types)
            fail_prob = random.uniform(0.1, 0.9)
            last_maintenance = datetime.now() - timedelta(days=random.randint(30, 365))
            
            equipment_data[f"{eq_type}_{i+1:03d}"] = {
                'type': eq_type,
                'location': random.choice(locations),
                'failure_probability': fail_prob,
                'risk_level': 'HIGH' if fail_prob > 0.7 else 'MEDIUM' if fail_prob > 0.4 else 'LOW',
                'next_maintenance': (datetime.now() + timedelta(days=random.randint(7, 90))).strftime('%Y-%m-%d'),
                'last_maintenance': last_maintenance.strftime('%Y-%m-%d'),
                'maintenance_cost': random.randint(200, 5000),
                'technician': random.choice(['John Smith', 'Sarah Johnson', 'Mike Chen', 'Lisa Rodriguez']),
                'priority': random.choice(['Critical', 'High', 'Medium', 'Low']),
                'estimated_hours': random.randint(1, 8)
            }
        return equipment_data

    def get_equipment_by_risk(self, risk_level: str) -> List[Dict]:
        return [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() 
                if edata['risk_level'] == risk_level.upper()]

    def get_maintenance_schedule(self, days_ahead: int = 30) -> List[Dict]:
        target_date = datetime.now() + timedelta(days=days_ahead)
        items = [{'id': eid, **edata} for eid, edata in self.maintenance_data.items() 
                if datetime.strptime(edata['next_maintenance'], '%Y-%m-%d') <= target_date]
        return sorted(items, key=lambda x: x['next_maintenance'])

    def get_statistics(self) -> Dict:
        total = len(self.maintenance_data)
        high_risk = len(self.get_equipment_by_risk('HIGH'))
        medium_risk = len(self.get_equipment_by_risk('MEDIUM'))
        low_risk = len(self.get_equipment_by_risk('LOW'))
        upcoming_week = len(self.get_maintenance_schedule(7))
        upcoming_month = len(self.get_maintenance_schedule(30))
        
        return {
            'total': total,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'upcoming_week': upcoming_week,
            'upcoming_month': upcoming_month
        }

# --- Enhanced Tool 1: Data Analysis Tool ---
def analyze_csv_data(query: str) -> str:
    try:
        # Find CSV files with various patterns
        csv_patterns = ["*ticket*.csv", "*jira*.csv", "*issue*.csv", "*.csv"]
        jira_files = []
        for pattern in csv_patterns:
            jira_files.extend(glob.glob(pattern, recursive=True))
        
        if not jira_files:
            return """
            <div class="alert alert-warning">
                <strong>⚠️ No CSV files found</strong><br>
                Please upload CSV files containing ticket or issue data to enable analysis.
            </div>
            """
        
        # Use the first available CSV file
        jira_file = jira_files[0]
        df = pd.read_csv(jira_file)
        
        # Enhanced query parsing
        query_lower = query.lower()
        patterns = [
            r"count of (.+?) issues?",
            r"how many (.+?) issues?",
            r"(.+?) issue count",
            r"related to a?n? '(.+?)' issue",
            r"(.+?) tickets?",
            r"number of (.+?) tickets?",
            r"total (.+?) incidents?"
        ]
        
        target_value = None
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                target_value = next(g for g in match.groups() if g is not None).strip()
                break
        
        if not target_value:
            available_cols = ", ".join(df.columns[:5])
            return f"""
            <div class="alert alert-info">
                <strong>🔍 Query unclear</strong><br>
                Found CSV file '{os.path.basename(jira_file)}' with {len(df)} records.<br>
                <strong>Available columns:</strong> {available_cols}<br>
                <strong>Try asking:</strong> "How many Device Faulty issues?" or "Count of Network tickets"
            </div>
            """
        
        # Enhanced column detection
        analysis_columns = ['root cause analysis', 'rca', 'issue type', 'category', 'problem type', 
                           'cause', 'classification', 'ticket type', 'incident type']
        target_column = None
        
        for col in df.columns:
            if any(ac in col.lower() for ac in analysis_columns):
                target_column = col
                break
        
        if not target_column:
            return f"""
            <div class="alert alert-error">
                <strong>❌ Analysis column not found</strong><br>
                Could not find a suitable column for analysis in '{os.path.basename(jira_file)}'.<br>
                <strong>Available columns:</strong> {', '.join(df.columns)}
            </div>
            """
        
        # Perform analysis
        matching_rows = df[df[target_column].str.contains(target_value, case=False, na=False)]
        count = len(matching_rows)
        total = len(df)
        percentage = (count / total * 100) if total > 0 else 0
        
        if count > 0:
            return f"""
            <div class="alert alert-success">
                <strong>✅ Analysis Results</strong><br>
                Found <strong>{count:,} tickets</strong> related to '<strong>{target_value}</strong>' 
                in the '{target_column}' column.<br>
                This represents <strong>{percentage:.1f}%</strong> of all {total:,} tickets.
            </div>
            """
        else:
            return f"""
            <div class="alert alert-warning">
                <strong>⚠️ No matches found</strong><br>
                No tickets found related to '<strong>{target_value}</strong>' in the '{target_column}' column.<br>
                File contains {total:,} total tickets.
            </div>
            """
            
    except Exception as e:
        return f"""
        <div class="alert alert-error">
            <strong>❌ Analysis Error</strong><br>
            {str(e)}<br>
            Please ensure the CSV file is properly formatted.
        </div>
        """

# --- Enhanced Maintenance Context Tool ---
def get_maintenance_context_with_actions(query: str, maintenance_pipeline: MaintenancePipeline, 
                                       search_func=None, search_args=None) -> str:
    query_lower = query.lower()
    context_parts = []
    
    if any(kw in query_lower for kw in ['risk', 'failure', 'alert', 'critical']):
        high_risk = maintenance_pipeline.get_equipment_by_risk('HIGH')
        medium_risk = maintenance_pipeline.get_equipment_by_risk('MEDIUM')
        
        if high_risk or medium_risk:
            context_parts.append("### 🚨 Equipment Risk Assessment")
            context_parts.append("")
            
            # High risk equipment
            if high_risk:
                context_parts.append("**Critical Risk Equipment:**")
                for eq in high_risk[:5]:
                    context_parts.append(
                        f"• **{eq['id']}** | {eq['location']} | "
                        f"Failure Risk: {eq['failure_probability']:.0%} | "
                        f"Next Service: {eq['next_maintenance']} | "
                        f"Cost: ${eq['maintenance_cost']:,}"
                    )
                    
                    if search_func and search_args:
                        action_query = f"troubleshooting procedure for {eq['type']}"
                        action_results = search_func(action_query, **search_args)
                        if action_results:
                            source = os.path.basename(action_results[0]['source'])
                            context_parts.append(f"  └─ **Action Guide:** From {source}")
                context_parts.append("")
            
            # Medium risk equipment summary
            if medium_risk:
                context_parts.append(f"**Medium Risk Items:** {len(medium_risk)} units require monitoring")
                context_parts.append("")
                
    elif any(kw in query_lower for kw in ['schedule', 'calendar', 'upcoming', 'maintenance']):
        schedule = maintenance_pipeline.get_maintenance_schedule(30)
        stats = maintenance_pipeline.get_statistics()
        
        if schedule:
            context_parts.append("### 📅 Maintenance Schedule Overview")
            context_parts.append("")
            context_parts.append(f"**Upcoming Tasks:** {len(schedule)} items in next 30 days")
            context_parts.append(f"**This Week:** {stats['upcoming_week']} urgent tasks")
            context_parts.append("")
            
            # Group by week
            week_1 = [item for item in schedule[:10] if 
                     (datetime.strptime(item['next_maintenance'], '%Y-%m-%d') - datetime.now()).days <= 7]
            
            if week_1:
                context_parts.append("**This Week's Priority Tasks:**")
                for item in week_1:
                    days_until = (datetime.strptime(item['next_maintenance'], '%Y-%m-%d') - datetime.now()).days
                    urgency = "🔥 URGENT" if days_until <= 2 else "⚠️ Soon" if days_until <= 5 else "📋 Scheduled"
                    context_parts.append(
                        f"• {urgency} | **{item['id']}** | {item['location']} | "
                        f"{item['next_maintenance']} | ${item['maintenance_cost']:,}"
                    )
        else:
            context_parts.append("### ✅ No maintenance scheduled for the next 30 days")
    
    return "\n".join(context_parts) if context_parts else ""

# --- Enhanced LLM Response Generation ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"🔑 Gemini API configuration error: {e}")
    GEMINI_MODEL = None

def generate_response_stream(query: str, chat_history: List[Dict], context: str) -> Generator:
    if not GEMINI_MODEL:
        yield "⚠️ AI model not configured. Please check your API key in Streamlit secrets."
        return
    
    # Enhanced prompt engineering
    history_summary = ""
    if chat_history:
        recent_history = chat_history[-4:]  # Last 4 messages for context
        history_summary = "\n".join([f"{msg['role'].title()}: {msg['content'][:200]}..." 
                                    for msg in recent_history])
    
    enhanced_prompt = f"""You are an expert AI support assistant with deep knowledge of IT systems, maintenance, and data analysis.

**Response Guidelines:**
- Provide clear, actionable answers
- Use specific data and metrics when available
- Include relevant recommendations
- Format responses for easy scanning (bullet points, sections)
- Maintain a professional but approachable tone
- When uncertain, clearly state limitations

**Recent Conversation Context:**
{history_summary or "This is a new conversation."}

**Knowledge Base Context:**
{context if context else "No specific context retrieved for this query."}

**User Question:** "{query}"

Please provide a comprehensive, helpful response based on the available information."""

    try:
        response_stream = GEMINI_MODEL.generate_content(enhanced_prompt, stream=True)
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"🔧 Response generation error: {str(e)}. Please try rephrasing your question."

# --- Enhanced Document Processing ---
def extract_text_from_pdf(file_path: str) -> str:
    try:
        text_content = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    text_content.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_content)
    except Exception as e:
        st.warning(f"📄 PDF processing error for {os.path.basename(file_path)}: {e}")
        return ""

def enhanced_chunking(text: str, chunk_size: int) -> List[str]:
    """Enhanced text chunking with better boundary detection"""
    # Split by paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # Further split long paragraphs by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    return [chunk for chunk in chunks if len(chunk) > 50]

@st.cache_resource(show_spinner=False)
def load_and_process_documents():
    """Enhanced document loading with better error handling and progress tracking"""
    
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf", "**/*.eml", "**/*.mbox"]
    all_files = []
    
    # Collect all files
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out system files and empty files
    valid_files = []
    for file_path in all_files:
        try:
            if os.path.getsize(file_path) > 100:  # At least 100 bytes
                valid_files.append(file_path)
        except:
            continue
    
    if not valid_files:
        return [], []
    
    docs, file_paths = [], []
    
    # Create progress tracking
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_path in enumerate(valid_files):
            file_name = os.path.basename(file_path)
            status_text.text(f"📁 Processing: {file_name}")
            progress_bar.progress((i + 1) / len(valid_files))
            
            content = ""
            try:
                if file_path.endswith('.pdf'):
                    content = extract_text_from_pdf(file_path)
                elif file_path.endswith('.eml'):
                    with open(file_path, 'rb') as f:
                        msg = BytesParser(policy=policy.default).parse(f)
                        content = format_email_for_rag(msg)
                elif file_path.endswith('.mbox'):
                    try:
                        mbox_emails = []
                        for msg in mailbox.mbox(file_path):
                            email_content = format_email_for_rag(msg)
                            if email_content:
                                mbox_emails.append(email_content)
                        content = "\n\n".join(mbox_emails)
                    except:
                        continue
                else:
                    # Text files
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                
                if content and len(content.strip()) > 100:
                    # Clean and validate content
                    cleaned_content = re.sub(r'\s+', ' ', content.strip())
                    docs.append(cleaned_content)
                    file_paths.append(file_path)
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_name}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
    
    if docs:
        st.success(f"✅ Successfully processed **{len(docs)}** documents from {len(valid_files)} files")
    else:
        st.warning("⚠️ No valid documents found to process")
        
    return docs, file_paths

def format_email_for_rag(msg) -> str:
    """Enhanced email formatting for RAG"""
    try:
        # Extract email body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    if "attachment" not in str(part.get("Content-Disposition", "")):
                        try:
                            body = part.get_payload(decode=True).decode(
                                part.get_content_charset() or 'utf-8', errors='ignore'
                            )
                            break
                        except:
                            continue
        else:
            if msg.get_content_type() == "text/plain":
                try:
                    body = msg.get_payload(decode=True).decode(
                        msg.get_content_charset() or 'utf-8', errors='ignore'
                    )
                except:
                    return ""
        
        if not body.strip():
            return ""
        
        # Format email with metadata
        email_data = {
            'from': msg.get('From', 'Unknown'),
            'to': msg.get('To', 'Unknown'),
            'subject': msg.get('Subject', 'No Subject'),
            'date': msg.get('Date', 'Unknown Date'),
            'body': body.strip()
        }
        
        return f"""--- EMAIL MESSAGE ---
From: {email_data['from']}
To: {email_data['to']}
Subject: {email_data['subject']}
Date: {email_data['date']}

{email_data['body']}
--- END EMAIL ---"""
        
    except Exception:
        return ""

@st.cache_resource(show_spinner=False)
def create_search_index(_documents, _file_paths):
    """Enhanced search index creation with better error handling"""
    if not _documents:
        return None, None, [], []
    
    try:
        # Load model with error handling
        with st.spinner("🔄 Loading AI model for document search..."):
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Process documents into chunks
        all_chunks = []
        chunk_metadata = []
        
        chunk_progress = st.progress(0)
        chunk_status = st.empty()
        
        for i, (doc, file_path) in enumerate(zip(_documents, _file_paths)):
            file_name = os.path.basename(file_path)
            chunk_status.text(f"📝 Chunking: {file_name}")
            
            chunks = enhanced_chunking(doc, config.chunk_size)
            all_chunks.extend(chunks)
            
            # Enhanced metadata
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'path': file_path,
                    'filename': file_name,
                    'chunk_index': chunk_idx,
                    'chunk_length': len(chunk),
                    'file_type': os.path.splitext(file_path)[1].lower()
                })
            
            chunk_progress.progress((i + 1) / len(_documents))
        
        chunk_progress.empty()
        chunk_status.empty()
        
        if not all_chunks:
            st.warning("⚠️ No valid text chunks created from documents")
            return None, None, [], []
        
        # Create embeddings
        with st.spinner(f"🧠 Creating embeddings for {len(all_chunks)} text chunks..."):
            embeddings = model.encode(
                all_chunks, 
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=32  # Optimize batch size
            )
        
        # Create FAISS index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        
        success_msg = f"✅ **Search index created successfully!**\n- {len(all_chunks):,} text chunks\n- {len(_documents)} documents\n- Ready for semantic search"
        st.success(success_msg)
        
        return index, model, all_chunks, chunk_metadata
        
    except Exception as e:
        st.error(f"❌ Error creating search index: {str(e)}")
        return None, None, [], []

def search_documents(query: str, index, model, chunks: List[str], metadata: List[Dict]) -> List[Dict]:
    """Enhanced document search with better ranking"""
    if not query or index is None:
        return []
    
    try:
        # Create query embedding
        query_embedding = model.encode([query], normalize_embeddings=True)
        
        # Search with increased results for reranking
        k = min(config.top_k_retrieval * 2, len(chunks))
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        results = []
        seen_files = set()
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and scores[0][i] > config.similarity_threshold:
                chunk_metadata = metadata[idx]
                
                # Prefer diverse sources (avoid too many results from same file)
                file_key = chunk_metadata['filename']
                if file_key in seen_files and len([r for r in results if r['filename'] == file_key]) >= 2:
                    continue
                    
                seen_files.add(file_key)
                
                results.append({
                    'content': chunks[idx],
                    'score': float(scores[0][i]),
                    'source': chunk_metadata['path'],
                    'filename': chunk_metadata['filename'],
                    'chunk_index': chunk_metadata['chunk_index'],
                    'file_type': chunk_metadata['file_type']
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:config.top_k_retrieval]
        
    except Exception as e:
        st.error(f"🔍 Search error: {str(e)}")
        return []

# --- Enhanced Visualization Components ---
def create_maintenance_dashboard(maintenance_pipeline: MaintenancePipeline):
    """Create interactive maintenance dashboard"""
    stats = maintenance_pipeline.get_statistics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card risk-high">
            <h3 style="margin:0; color: #F44336;">🚨 High Risk</h3>
            <h2 style="margin:5px 0;">{}</h2>
            <small>Critical Equipment</small>
        </div>
        """.format(stats['high_risk']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card risk-medium">
            <h3 style="margin:0; color: #FF9800;">⚠️ Medium Risk</h3>
            <h2 style="margin:5px 0;">{}</h2>
            <small>Monitoring Required</small>
        </div>
        """.format(stats['medium_risk']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card risk-low">
            <h3 style="margin:0; color: #4CAF50;">✅ Low Risk</h3>
            <h2 style="margin:5px 0;">{}</h2>
            <small>Stable Equipment</small>
        </div>
        """.format(stats['low_risk']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; color: #2196F3;">📅 This Week</h3>
            <h2 style="margin:5px 0;">{}</h2>
            <small>Scheduled Tasks</small>
        </div>
        """.format(stats['upcoming_week']), unsafe_allow_html=True)

def create_risk_distribution_chart(maintenance_pipeline: MaintenancePipeline):
    """Create risk distribution visualization"""
    stats = maintenance_pipeline.get_statistics()
    
    # Risk distribution pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['High Risk', 'Medium Risk', 'Low Risk'],
        values=[stats['high_risk'], stats['medium_risk'], stats['low_risk']],
        hole=0.4,
        marker_colors=['#F44336', '#FF9800', '#4CAF50']
    )])
    
    fig_pie.update_layout(
        title="Equipment Risk Distribution",
        showlegend=True,
        height=400,
        font=dict(size=14)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

def create_maintenance_timeline(maintenance_pipeline: MaintenancePipeline):
    """Create maintenance timeline visualization"""
    schedule = maintenance_pipeline.get_maintenance_schedule(60)
    
    if not schedule:
        st.info("📅 No maintenance scheduled in the next 60 days")
        return
    
    # Prepare data for timeline
    df_schedule = pd.DataFrame(schedule)
    df_schedule['next_maintenance'] = pd.to_datetime(df_schedule['next_maintenance'])
    df_schedule['days_until'] = (df_schedule['next_maintenance'] - pd.Timestamp.now()).dt.days
    
    # Create timeline chart
    fig_timeline = px.scatter(
        df_schedule.head(20), 
        x='next_maintenance', 
        y='type',
        size='maintenance_cost',
        color='risk_level',
        hover_data=['id', 'location', 'technician'],
        title="Upcoming Maintenance Timeline (Next 60 Days)",
        color_discrete_map={
            'HIGH': '#F44336',
            'MEDIUM': '#FF9800', 
            'LOW': '#4CAF50'
        }
    )
    
    fig_timeline.update_layout(
        height=500,
        xaxis_title="Maintenance Date",
        yaxis_title="Equipment Type"
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)

# --- Main Application ---
def main():
    st.title("🤖 AI Support Assistant")
    st.markdown("### Intelligent Document Search • Predictive Maintenance • Data Analysis")
    
    # Initialize maintenance pipeline
    if 'maintenance_pipeline' not in st.session_state:
        st.session_state.maintenance_pipeline = MaintenancePipeline()
    
    maintenance_pipeline = st.session_state.maintenance_pipeline
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.settings["chunk_size"] = st.slider(
                "Chunk Size", 200, 1000, st.session_state.settings["chunk_size"], 50
            )
            st.session_state.settings["show_sources"] = st.checkbox(
                "Show Sources", st.session_state.settings["show_sources"]
            )
            st.session_state.settings["auto_scroll"] = st.checkbox(
                "Auto Scroll", st.session_state.settings["auto_scroll"]
            )
        
        # Quick Actions
        st.header("🚀 Quick Actions")
        
        if st.button("📊 View Maintenance Dashboard", use_container_width=True):
            st.session_state.show_dashboard = True
        
        if st.button("📈 Risk Analysis", use_container_width=True):
            st.session_state.show_risk_analysis = True
        
        if st.button("📅 Maintenance Schedule", use_container_width=True):
            st.session_state.show_schedule = True
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Document status
        st.header("📚 Document Status")
        with st.spinner("🔄 Loading documents..."):
            docs, file_paths = load_and_process_documents()
        
        if docs:
            st.success(f"✅ {len(docs)} documents loaded")
            
            # Create search index
            with st.spinner("🧠 Building search index..."):
                index, model, chunks, metadata = create_search_index(docs, file_paths)
                
            if index is not None:
                st.success(f"🔍 Search ready ({len(chunks)} chunks)")
            else:
                st.error("❌ Search index failed")
                index, model, chunks, metadata = None, None, [], []
        else:
            st.warning("⚠️ No documents found")
            index, model, chunks, metadata = None, None, [], []
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Assistant", "📊 Maintenance Dashboard", "📈 Analytics", "🔧 Tools"])
    
    with tab1:
        # Chat interface
        st.header("💬 AI Assistant Chat")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>{message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_query = st.text_input(
                    "Ask me anything about your documents, maintenance, or data analysis:",
                    placeholder="e.g., 'Show high-risk equipment' or 'How many network issues?'"
                )
            with col2:
                submit_button = st.form_submit_button("Send 🚀", use_container_width=True)
        
        if submit_button and user_query:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Process query
            with st.spinner("🤔 Thinking..."):
                # Search documents
                search_results = []
                if index is not None:
                    search_results = search_documents(user_query, index, model, chunks, metadata)
                
                # Analyze CSV data
                csv_analysis = analyze_csv_data(user_query)
                
                # Get maintenance context
                maintenance_context = get_maintenance_context_with_actions(
                    user_query, maintenance_pipeline, search_documents, 
                    {'index': index, 'model': model, 'chunks': chunks, 'metadata': metadata}
                )
                
                # Combine context
                context_parts = []
                
                if search_results:
                    context_parts.append("=== DOCUMENT SEARCH RESULTS ===")
                    for i, result in enumerate(search_results[:3], 1):
                        source_info = f"Source: {result['filename']}"
                        context_parts.append(f"\n[Result {i}] {source_info}\n{result['content'][:500]}...")
                
                if csv_analysis and "No CSV files found" not in csv_analysis:
                    context_parts.append(f"\n=== DATA ANALYSIS ===\n{csv_analysis}")
                
                if maintenance_context:
                    context_parts.append(f"\n=== MAINTENANCE INFO ===\n{maintenance_context}")
                
                combined_context = "\n".join(context_parts)
                
                # Generate response
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in generate_response_stream(user_query, st.session_state.chat_history, combined_context):
                    full_response += chunk
                    response_placeholder.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>{full_response}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # Show sources if enabled
                if st.session_state.settings["show_sources"] and search_results:
                    with st.expander("📚 Sources Used", expanded=False):
                        for result in search_results:
                            st.markdown(f"""
                            **{result['filename']}** (Score: {result['score']:.3f})
                            > {result['content'][:200]}...
                            """)
            
            # Auto-scroll if enabled
            if st.session_state.settings["auto_scroll"]:
                st.rerun()
    
    with tab2:
        st.header("📊 Maintenance Dashboard")
        create_maintenance_dashboard(maintenance_pipeline)
        
        col1, col2 = st.columns(2)
        with col1:
            create_risk_distribution_chart(maintenance_pipeline)
        
        with col2:
            # Equipment by type
            equipment_data = maintenance_pipeline.maintenance_data
            type_counts = {}
            for eq_data in equipment_data.values():
                eq_type = eq_data['type']
                type_counts[eq_type] = type_counts.get(eq_type, 0) + 1
            
            fig_bar = px.bar(
                x=list(type_counts.keys()),
                y=list(type_counts.values()),
                title="Equipment Count by Type",
                color=list(type_counts.values()),
                color_continuous_scale="viridis"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline
        create_maintenance_timeline(maintenance_pipeline)
    
    with tab3:
        st.header("📈 Data Analytics")
        
        # High-risk equipment details
        high_risk_equipment = maintenance_pipeline.get_equipment_by_risk('HIGH')
        if high_risk_equipment:
            st.subheader("🚨 High-Risk Equipment Details")
            df_high_risk = pd.DataFrame(high_risk_equipment)
            st.dataframe(
                df_high_risk[['id', 'type', 'location', 'failure_probability', 'next_maintenance', 'maintenance_cost']],
                use_container_width=True
            )
        
        # Cost analysis
        st.subheader("💰 Maintenance Cost Analysis")
        equipment_data = list(maintenance_pipeline.maintenance_data.values())
        df_equipment = pd.DataFrame(equipment_data)
        
        # Cost by type
        cost_by_type = df_equipment.groupby('type')['maintenance_cost'].sum().reset_index()
        fig_cost = px.pie(
            cost_by_type, 
            values='maintenance_cost', 
            names='type',
            title="Maintenance Cost Distribution by Equipment Type"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
        
        # Cost vs Risk scatter
        fig_scatter = px.scatter(
            df_equipment,
            x='failure_probability',
            y='maintenance_cost',
            color='risk_level',
            size='estimated_hours',
            hover_data=['type', 'location'],
            title="Maintenance Cost vs Failure Probability",
            color_discrete_map={
                'HIGH': '#F44336',
                'MEDIUM': '#FF9800',
                'LOW': '#4CAF50'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab4:
        st.header("🔧 Tools & Utilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 CSV Data Analysis")
            sample_queries = [
                "How many device faulty issues?",
                "Count of network tickets",
                "Number of high priority incidents",
                "Total software issues"
            ]
            
            selected_query = st.selectbox("Sample Queries:", [""] + sample_queries)
            custom_query = st.text_input("Or enter custom query:")
            
            analysis_query = custom_query if custom_query else selected_query
            
            if st.button("Analyze Data") and analysis_query:
                result = analyze_csv_data(analysis_query)
                st.markdown(result, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔍 Document Search")
            search_query = st.text_input("Search documents:")
            
            if st.button("Search") and search_query and index is not None:
                results = search_documents(search_query, index, model, chunks, metadata)
                
                if results:
                    st.success(f"Found {len(results)} relevant results:")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}: {result['filename']} (Score: {result['score']:.3f})"):
                            st.text(result['content'][:500] + "...")
                else:
                    st.warning("No relevant documents found.")
        
        # System information
        st.subheader("ℹ️ System Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(docs) if docs else 0)
        
        with col2:
            st.metric("Search Chunks", len(chunks) if chunks else 0)
        
        with col3:
            st.metric("Total Equipment", len(maintenance_pipeline.maintenance_data))

if __name__ == "__main__":
    main()
