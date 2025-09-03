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
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

# --- Simple but Effective RAG Configuration ---
class RAGConfig:
    def __init__(self):
        self.chunk_size = 400
        self.chunk_overlap = 50
        self.top_k_retrieval = 8
        self.similarity_threshold = 0.1  # Lower threshold to get more results
        self.max_context_length = 2000

config = RAGConfig()

# --- Simple Text Processing ---
def clean_text(text: str) -> str:
    """Simple but effective text cleaning"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def simple_chunking(text: str, chunk_size: int = 400) -> List[str]:
    """Simple but effective chunking"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only add substantial chunks
            chunks.append(chunk.strip())
    
    return chunks

def detect_query_type(query: str) -> str:
    """Simple query type detection"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['camera', 'video', 'recording', 'webcam']):
        return 'camera'
    elif any(word in query_lower for word in ['audio', 'microphone', 'speaker', 'sound', 'mic']):
        return 'audio'
    elif any(word in query_lower for word in ['laptop', 'computer', 'pc', 'workstation']):
        return 'computer'
    elif any(word in query_lower for word in ['projector', 'display', 'monitor', 'screen']):
        return 'display'
    elif any(word in query_lower for word in ['issue', 'problem', 'error', 'bug', 'ticket']):
        return 'issue'
    elif any(word in query_lower for word in ['all', 'everything', 'complete', 'full', 'entire']):
        return 'comprehensive'
    else:
        return 'general'

# --- Document Processing ---
@st.cache_resource
def load_documents_simple():
    """Load documents with simple but effective processing"""
    st.info("🔍 Loading documents...")
    
    documents = []
    file_paths = []
    file_metadata = []
    
    # Look for files
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv"]
    
    for pattern in file_patterns:
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                file_name = os.path.basename(file_path)
                
                if file_path.endswith('.csv'):
                    # Handle CSV files
                    try:
                        df = pd.read_csv(file_path, encoding='latin-1')
                        
                        # Create a readable content representation
                        content = f"Equipment Data from {file_name}:\n\n"
                        content += f"Total items: {len(df)}\n"
                        content += f"Columns: {', '.join(df.columns)}\n\n"
                        
                        # Add all data in a searchable format
                        for idx, row in df.head(50).iterrows():  # First 50 rows
                            row_text = ""
                            for col, val in row.items():
                                if pd.notna(val):
                                    row_text += f"{col}: {val} | "
                            content += f"Item {idx + 1}: {row_text}\n"
                        
                        st.success(f"✅ CSV Loaded: {file_name} ({len(df)} rows)")
                        
                    except Exception as e:
                        # Fallback: read as text
                        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            content = f"Data from {file_name}:\n\n" + f.read()
                        st.warning(f"⚠️ CSV read as text: {file_name}")
                        
                else:
                    # Handle text files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    st.success(f"✅ Text Loaded: {file_name}")
                
                if content and len(content.strip()) > 20:
                    documents.append(clean_text(content))
                    file_paths.append(file_path)
                    file_metadata.append({
                        'name': file_name,
                        'path': file_path,
                        'type': 'csv' if file_path.endswith('.csv') else 'text',
                        'size': len(content)
                    })
                
            except Exception as e:
                st.error(f"❌ Error loading {file_path}: {str(e)}")
    
    st.info(f"📚 Total documents loaded: {len(documents)}")
    return documents, file_paths, file_metadata

@st.cache_resource
def create_search_index(_documents, _file_paths, _metadata):
    """Create search index with better chunking"""
    if not _documents:
        return None, None, None, None, None
    
    st.info("🔨 Creating search index...")
    
    all_chunks = []
    chunk_sources = []
    chunk_metadata = []
    
    for i, (doc, file_path, metadata) in enumerate(zip(_documents, _file_paths, _metadata)):
        chunks = simple_chunking(doc, config.chunk_size)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_sources.append(file_path)
            chunk_metadata.append(metadata)
    
    st.info(f"📄 Created {len(all_chunks)} searchable chunks")
    
    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lighter, faster model
    
    st.info("🧠 Generating embeddings...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=16)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    st.success("✅ Search index created successfully!")
    
    return index, all_chunks, chunk_sources, chunk_metadata, model

def search_documents(index, chunks, chunk_sources, chunk_metadata, model, query: str, k: int = 8):
    """Simple but effective search"""
    if index is None or not chunks:
        return []
    
    try:
        # Create query embedding
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding.astype('float32'), min(k, len(chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and scores[0][i] > config.similarity_threshold:
                results.append({
                    'content': chunks[idx],
                    'source': chunk_sources[idx],
                    'metadata': chunk_metadata[idx],
                    'score': float(scores[0][i]),
                    'rank': i + 1
                })
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# --- Response Generation ---
def generate_response(query: str, search_results: List[Dict]) -> str:
    """Generate response based on search results"""
    
    if not search_results:
        return f"""I couldn't find specific information about "{query}" in your documents.

**What I can help you find:**
• Equipment information (cameras, audio, computers)
• Issue tracking and tickets
• Project documentation
• Technical specifications

**Try asking:**
• "Show me all equipment"
• "What cameras are available?"
• "List audio equipment"
• "What's in the inventory?"

**Loaded files:** {', '.join([meta.get('name', 'Unknown') for _, _, meta, _, _ in zip([], [], file_metadata, [], []) if 'file_metadata' in globals()])}"""

    query_type = detect_query_type(query)
    
    # Start response
    response = f"Here's what I found for your query about **{query}**:\n\n"
    
    # Group results by source
    results_by_source = {}
    for result in search_results[:6]:  # Top 6 results
        source_name = os.path.basename(result['source'])
        if source_name not in results_by_source:
            results_by_source[source_name] = []
        results_by_source[source_name].append(result)
    
    # Generate response for each source
    for source_name, source_results in results_by_source.items():
        response += f"### 📄 From {source_name}\n\n"
        
        # Show relevant content from this source
        shown_content = set()  # Avoid duplicates
        item_count = 0
        
        for result in source_results:
            if item_count >= 5:  # Max 5 items per source
                break
                
            content = result['content']
            
            # For CSV data, extract individual items
            if 'csv' in source_name.lower() and 'Item' in content:
                items = re.findall(r'Item \d+: (.+)', content)
                for item in items[:5]:
                    if item not in shown_content and len(item.strip()) > 20:
                        # Clean up the item display
                        clean_item = item.replace(' | ', ' • ').strip(' •')
                        response += f"• {clean_item}\n"
                        shown_content.add(item)
                        item_count += 1
                        if item_count >= 5:
                            break
            else:
                # For other content, show relevant snippets
                if content[:100] not in shown_content:
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    response += f"• {content_preview}\n"
                    shown_content.add(content[:100])
                    item_count += 1
        
        response += "\n"
    
    # Add summary
    response += f"---\n\n**Found {len(search_results)} relevant results** from {len(results_by_source)} source(s)\n\n"
    
    # Add suggestions based on query type
    if query_type == 'camera':
        response += "💡 **Related queries:** Try asking about 'video equipment' or 'recording devices'\n"
    elif query_type == 'audio':
        response += "💡 **Related queries:** Try asking about 'microphones' or 'sound systems'\n"
    elif query_type == 'comprehensive':
        response += "💡 **Tip:** For specific categories, try 'cameras', 'audio equipment', or 'computers'\n"
    else:
        response += "💡 **Tip:** Be more specific for better results (e.g., 'Canon cameras' or 'wireless microphones')\n"
    
    return response

# --- Main Streamlit App ---
st.set_page_config(page_title="Simple Smart Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 Simple Smart Support Chatbot")
st.markdown("*Direct access to your documents - no complex processing, just results!*")

# Load data
with st.spinner("Loading your documents..."):
    documents, file_paths, file_metadata = load_documents_simple()
    
    if not documents:
        st.error("❌ No documents found!")
        st.info("""
        **Add files to your repository:**
        - CSV files (equipment lists, inventory)
        - Text files (.txt, .md)
        - Place them in the same directory as this script
        """)
        st.stop()

# Create search index
with st.spinner("Building search capabilities..."):
    index, chunks, chunk_sources, chunk_metadata, model = create_search_index(
        documents, file_paths, file_metadata
    )
    
    if index is None:
        st.error("❌ Failed to create search index")
        st.stop()

st.success("🚀 Chatbot ready! Your documents are loaded and searchable.")

# Sidebar info
with st.sidebar:
    st.header("📊 Loaded Data")
    for meta in file_metadata:
        st.write(f"📄 **{meta['name']}** ({meta['type']}, {meta['size']:,} chars)")
    
    st.header("🔧 Configuration")
    st.write(f"🔍 Chunks: {len(chunks) if 'chunks' in locals() else 0}")
    st.write(f"🎯 Search Results: {config.top_k_retrieval}")
    st.write(f"📏 Similarity Threshold: {config.similarity_threshold}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Welcome message
    welcome = f"""👋 **Welcome!** I've loaded {len(documents)} document(s) and I'm ready to help!

**What I can do:**
• Find equipment in your inventory
• Search through your documents
• Answer questions about your data

**Your loaded files:**
{chr(10).join([f"• {meta['name']}" for meta in file_metadata])}

**Try asking:**
• "Show me all cameras"
• "What audio equipment do we have?"
• "List everything in the inventory"
• "Find projectors"

What would you like to know?"""
    
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your equipment and documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching through your documents..."):
            # Search for relevant information
            search_results = search_documents(
                index, chunks, chunk_sources, chunk_metadata, model, prompt, config.top_k_retrieval
            )
            
            # Generate response
            response = generate_response(prompt, search_results)
            st.markdown(response)
            
            # Debug info
            if search_results:
                with st.expander("🔍 Search Details"):
                    st.write(f"**Query:** {prompt}")
                    st.write(f"**Results found:** {len(search_results)}")
                    for i, result in enumerate(search_results[:3]):
                        st.write(f"**Result {i+1}:** Score: {result['score']:.3f}, Source: {os.path.basename(result['source'])}")
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
