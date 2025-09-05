import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
from typing import List, Dict, Any, Generator, Optional
import fitz  # PyMuPDF
from datetime import datetime, timedelta
import random
from dataclasses import dataclass
import google.generativeai as genai
import mailbox
from email import policy
from email.parser import BytesParser
import markdown
import queue
from pathlib import Path
import time
import html

# --- Advanced Configuration ---
@dataclass
class AppConfig:
    chunk_size: int = 500
    top_k_retrieval: int = 5
    similarity_threshold: float = 0.4
    max_retries: int = 3
    retry_delay: float = 1.0
    message_queue_size: int = 100
    max_history_tokens: int = 2000
    cache_dir: Path = Path("./cache")
    
config = AppConfig()

# --- Enhanced UI Components ---
class ChatUI:
    @staticmethod
    def render_message(content: str, is_user: bool, timestamp: str, 
                      enable_reactions: bool = True) -> None:
        theme = st.session_state.settings.get("theme", "light")
        colors = {
            "light": {
                "user": "#DCF8C6",
                "assistant": "#F1F0F0",
                "text": "#222",
                "border": "#ddd"
            },
            "dark": {
                "user": "#3A6351",
                "assistant": "#222831",
                "text": "#fff", 
                "border": "#393E46"
            }
        }
        
        c = colors[theme]
        color = c["user"] if is_user else c["assistant"]
        
        # Process markdown and code blocks
        if not is_user:
            content = markdown.markdown(
                content,
                extensions=['fenced_code', 'tables']
            )
            
            # Add copy buttons to code blocks
            content = re.sub(
                r'<pre><code>(.*?)</code></pre>',
                lambda m: f'''
                <div class="code-block">
                    <button onclick="navigator.clipboard.writeText(`{html.escape(m.group(1))}`)">
                        📋 Copy
                    </button>
                    <pre><code>{m.group(1)}</code></pre>
                </div>''',
                content,
                flags=re.DOTALL
            )
        
        st.markdown(
            f"""
            <div class="chat-message {'user' if is_user else 'assistant'}"
                 style="display:flex;flex-direction:{'row-reverse' if is_user else 'row'};
                        align-items:flex-start;margin:12px 0;">
                <img src="{'https://cdn-icons-png.flaticon.com/512/1946/1946429.png' if is_user 
                          else 'https://cdn-icons-png.flaticon.com/512/4712/4712035.png'}"
                     style="width:40px;height:40px;border-radius:50%;margin:0 8px;">
                     
                <div style="background:{color};color:{c['text']};
                            padding:16px;border-radius:12px;
                            max-width:75%;border:1px solid {c['border']};
                            box-shadow:0 2px 8px rgba(0,0,0,0.07);">
                    <div style="font-weight:500;margin-bottom:4px;">
                        {f"You" if is_user else "AI Assistant"}
                    </div>
                    <div style="margin:8px 0;">
                        {content}
                    </div>
                    <div style="font-size:11px;opacity:0.7;text-align:right;">
                        {timestamp}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if enable_reactions and not is_user:
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("👍", key=f"like_{timestamp}"):
                    st.success("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"dislike_{timestamp}"):
                    st.text_input("What could be improved?", key=f"feedback_{timestamp}")

# --- Enhanced Message Management ---
class MessageManager:
    def __init__(self, max_size: int = 100):
        self.queue = queue.Queue(maxsize=max_size)
        self.history: List[Dict] = []
        
    def add_message(self, role: str, content: str) -> None:
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        self.history.append(msg)
        self.queue.put(msg)
        
    def get_recent_history(self, max_tokens: int = 2000) -> str:
        history = []
        token_count = 0
        for msg in reversed(self.history):
            tokens = len(msg["content"].split())
            if token_count + tokens > max_tokens:
                break
            history.append(f"{msg['role'].title()}: {msg['content']}")
            token_count += tokens
        return "\n".join(reversed(history))

# --- Document Processing with Caching ---
class DocumentProcessor:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def process_document(self, file_path: str) -> Optional[str]:
        cache_key = str(hash(file_path))
        cache_file = self.cache_dir / f"{cache_key}.txt"
        
        if cache_file.exists():
            return cache_file.read_text(encoding='utf-8')
            
        content = ""
        try:
            if file_path.endswith('.pdf'):
                content = self._process_pdf(file_path)
            elif file_path.endswith('.eml'):
                content = self._process_email(file_path)
            elif file_path.endswith('.mbox'):
                content = self._process_mbox(file_path)
            else:
                content = self._process_text(file_path)
                
            if content:
                cache_file.write_text(content, encoding='utf-8')
            return content
        except Exception as e:
            st.error(f"Error processing {os.path.basename(file_path)}: {e}")
            return None
            
    def _process_pdf(self, path: str) -> str:
        with fitz.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
            
    def _process_email(self, path: str) -> str:
        with open(path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
            return format_email_for_rag(msg)
            
    def _process_mbox(self, path: str) -> str:
        messages = [format_email_for_rag(msg) for msg in mailbox.mbox(path)]
        return "\n\n".join(filter(None, messages))
        
    def _process_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()

# --- Initialize Components ---
@st.cache_resource
def init_components():
    msg_manager = MessageManager(config.message_queue_size)
    doc_processor = DocumentProcessor(config.cache_dir)
    maintenance = MaintenancePipeline()
    return msg_manager, doc_processor, maintenance

# --- Main App UI ---
def main():
    st.set_page_config(
        page_title="AI Support Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    msg_manager, doc_processor, maintenance = init_components()
    
    # Custom CSS
    st.markdown("""
        <style>
        .code-block { position: relative; }
        .code-block button {
            position: absolute;
            top: 5px;
            right: 5px;
            background: #666;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
        }
        .chat-message img { transition: transform 0.2s; }
        .chat-message img:hover { transform: scale(1.1); }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Support Assistant")
        
        # Quick Actions
        st.markdown("### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Risk Alerts"):
                handle_quick_action("alerts", msg_manager, maintenance)
        with col2:
            if st.button("📅 Maintenance"):
                handle_quick_action("maintenance", msg_manager, maintenance)
                
        # System Status
        st.markdown("### System Status")
        high_risk = len(maintenance.get_equipment_by_risk('HIGH'))
        due_tasks = len(maintenance.get_maintenance_schedule(7))
        st.metric("High Risk Items", f"🚨 {high_risk}")
        st.metric("Tasks Due", f"📅 {due_tasks}")
        
        # Document Upload
        st.markdown("### Document Upload")
        uploaded_files = st.file_uploader(
            "Add new documents",
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                file_path = Path(file.name)
                file_path.write_bytes(file.getbuffer())
                if doc_processor.process_document(str(file_path)):
                    st.success(f"Processed: {file.name}")
                    
        # Settings
        st.markdown("### Settings")
        st.session_state.settings = st.session_state.get("settings", {
            "theme": "light",
            "chunk_size": 500
        })
        theme = st.radio(
            "Theme",
            ["light", "dark"],
            index=0 if st.session_state.settings["theme"]=="light" else 1
        )
        chunk_size = st.slider(
            "Context Window",
            200, 1000,
            st.session_state.settings["chunk_size"]
        )
        st.session_state.settings.update({
            "theme": theme,
            "chunk_size": chunk_size
        })
    
    # Chat Interface
    st.title("💬 AI Support Assistant")
    
    # Display Message History
    for msg in msg_manager.history:
        ChatUI.render_message(
            msg["content"],
            msg["role"] == "user",
            msg["timestamp"]
        )
    
    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        msg_manager.add_message("user", prompt)
        
        with st.spinner("Processing..."):
            try:
                # Generate context and response
                context = generate_context(prompt, maintenance, doc_processor)
                response = generate_response(
                    prompt,
                    context,
                    msg_manager.get_recent_history()
                )
                
                # Add AI response
                msg_manager.add_message("assistant", response)
                
                # Force refresh
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
