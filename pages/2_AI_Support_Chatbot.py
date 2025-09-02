# pages/2_AI_Support_Chatbot.py
import streamlit as st
import pandas as pd
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings
import re
warnings.filterwarnings("ignore")

# --- Helper Functions ---
def clean_and_format_text(text):
    """Clean and format text for better readability"""
    # Remove extra whitespace and clean up
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def generate_friendly_response(query, results):
    """Generate a friendly, conversational response based on search results"""
    if not results:
        return """I'd be happy to help you, but I couldn't find specific information about that in your documents.

Here are a few things you can try:
- Rephrase your question with different keywords
- Ask about specific topics I might have data on
- Check if the relevant documents are uploaded

What else would you like to know? 😊"""

    query_lower = query.lower()
    
    if any(word in query_lower for word in ['computer', 'laptop', 'pc', 'hardware']):
        intro = "Based on your documents, here are some computer/hardware related options I found:"
    elif any(word in query_lower for word in ['camera', 'video', 'recording']):
        intro = "For cameras and video equipment, I found these options in your data:"
    elif any(word in query_lower for word in ['audio', 'microphone', 'speaker', 'sound']):
        intro = "Here are the audio equipment options from your documents:"
    elif any(word in query_lower for word in ['issue', 'problem', 'error', 'bug', 'ticket']):
        intro = "I found some relevant information about issues or tickets:"
    elif any(word in query_lower for word in ['meeting', 'conference', 'call']):
        intro = "Regarding meetings and conferencing, here's what I found:"
    else:
        intro = "Based on your query, here's the most relevant information I found:"
    
    response = f"{intro}\n\n"
    
    for i, result in enumerate(results, 1):
        source_name = result['source']
        content = result['content']
        
        if 'csv' in source_name.lower():
            lines = content.split('\n')
            formatted_content = ""
            for line in lines[:3]:
                if line.strip() and not line.startswith('Unnamed'):
                    formatted_content += f"• {line.strip()}\n"
            response += f"**From {source_name}:**\n{formatted_content}\n"
        else:
            clean_content = clean_and_format_text(content)
            if len(clean_content) > 200:
                clean_content = clean_content[:200] + "..."
            response += f"**From {source_name}:**\n{clean_content}\n\n"
    
    response += "\n💡 **Need more details?** Feel free to ask more specific questions about any of these items!"
    return response

# --- Main Functions ---
@st.cache_resource
def load_documents():
    """Load and process documents from the repository"""
    documents = []
    file_paths = []
    
    file_patterns = {
        "**/*.txt": "text",
        "**/*.md": "text", 
        "**/*.csv": "csv"
    }
    
    for pattern, file_type in file_patterns.items():
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                if file_type == "csv":
                    # --- FIX IS HERE: Added encoding='latin-1' ---
                    df = pd.read_csv(file_path, encoding='latin-1')
                    content = f"Data from {os.path.basename(file_path)}:\n"
                    content += df.to_string(index=False)
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                if content.strip():
                    documents.append(content)
                    file_paths.append(file_path)
                    st.success(f"✅ Loaded: {os.path.basename(file_path)}")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_path}: {str(e)}")
    
    return documents, file_paths

@st.cache_resource
def create_search_index(_documents, _file_paths):
    """Create FAISS search index"""
    if not _documents:
        return None, None, None, None
    
    chunks = []
    chunk_sources = []
    
    for i, doc in enumerate(_documents):
        if len(doc) <= 1000:
            chunks.append(doc)
            chunk_sources.append(_file_paths[i])
        else:
            words = doc.split()
            current_chunk = []
            for word in words:
                current_chunk.append(word)
                if len(' '.join(current_chunk)) > 800:
                    chunks.append(' '.join(current_chunk))
                    chunk_sources.append(_file_paths[i])
                    current_chunk = []
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                chunk_sources.append(_file_paths[i])
    
    st.info(f"📄 Created {len(chunks)} searchable chunks")
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    return index, chunks, chunk_sources, model

def search_knowledge_base(index, chunks, chunk_sources, model, query, k=3):
    """Search the knowledge base and return formatted results"""
    if index is None:
        return []
    
    try:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and scores[0][i] > 0.1:
                results.append({
                    'content': chunks[idx],
                    'source': chunk_sources[idx],
                    'score': float(scores[0][i])
                })
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# --- Streamlit Interface ---
st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Support Assistant")
st.markdown("*Your friendly helper for finding information in your documents and JIRA tickets*")

with st.spinner("🔄 Setting up your AI assistant..."):
    documents, file_paths = load_documents()
    if not documents:
        st.error("❌ No documents found to work with!")
        st.markdown("""
        **Please add some files to your repository:**
        - Text files (.txt)
        - Markdown files (.md) 
        - CSV files with your JIRA ticket data (.csv)
        """)
        st.stop()

with st.spinner("🧠 Building search capabilities..."):
    index, chunks, chunk_sources, model = create_search_index(documents, file_paths)
    if index is None:
        st.error("❌ Couldn't create search index")
        st.stop()

st.success("✅ Your AI assistant is ready to help!")

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = """Hello! 👋 I'm your AI Support Assistant. 

I can help you find information from your documents and JIRA tickets. Here are some things you can ask me:

• **Equipment questions**: "What cameras do we have?" or "Show me audio equipment"
• **Issue tracking**: "What were the network issues?" or "Show me recent tickets"
• **Project info**: "Tell me about the Mumbai project" or "What are the open issues?"

What would you like to know?"""
    
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your projects, issues, or equipment..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Let me search through your documents..."):
            search_results = search_knowledge_base(index, chunks, chunk_sources, model, prompt)
            response = generate_friendly_response(prompt, search_results)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("📊 Assistant Status")
    if 'documents' in locals():
        st.metric("📄 Documents", len(documents))
    if 'chunks' in locals():
        st.metric("🔍 Search Chunks", len(chunks))
    
    st.header("💡 Tips")
    st.markdown("""
    **Ask me about:**
    - Equipment and inventory
    - JIRA tickets and issues  
    - Project status and updates
    - Technical problems
    - Meeting notes and decisions
    
    **Example queries:**
    - "What audio issues were reported?"
    - "Show me camera options"
    - "Any network problems recently?"
    """)
    
    if st.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()
        
    st.header("📁 Loaded Files")
    if 'file_paths' in locals():
        for fp in file_paths:
            st.write(f"📄 {os.path.basename(fp)}")
