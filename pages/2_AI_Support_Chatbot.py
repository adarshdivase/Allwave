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
import PyPDF2
import fitz  # PyMuPDF - better PDF handling
from io import BytesIO

warnings.filterwarnings("ignore")

# --- Enhanced RAG Configuration ---
class RAGConfig:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 75
        self.top_k_retrieval = 6
        self.similarity_threshold = 0.15
        self.max_context_length = 2500

config = RAGConfig()

# --- Enhanced Text Processing ---
def clean_text(text: str) -> str:
    """Enhanced text cleaning with better formatting preservation"""
    # Remove excessive whitespace but preserve line breaks for readability
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might interfere with display
    text = re.sub(r'[^\w\s\-.,;:()[\]{}!?@#$%^&*+=<>/"\'`~|\\]', ' ', text)
    # Clean up multiple spaces again
    text = re.sub(r'\s+', ' ', text)
    return text

def smart_chunking(text: str, chunk_size: int = 500) -> List[str]:
    """Smarter chunking that preserves sentence boundaries"""
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
    return chunks

def detect_query_intent(query: str) -> Dict[str, Any]:
    """Enhanced query analysis with intent detection"""
    query_lower = query.lower()
    
    # Define keywords for different categories
    categories = {
        'camera': ['camera', 'video', 'recording', 'webcam', 'camcorder', 'lens', 'canon', 'nikon', 'sony'],
        'audio': ['audio', 'microphone', 'speaker', 'sound', 'mic', 'headphone', 'amplifier'],
        'computer': ['laptop', 'computer', 'pc', 'workstation', 'desktop', 'server', 'tablet'],
        'display': ['projector', 'display', 'monitor', 'screen', 'tv', 'television'],
        'network': ['network', 'wifi', 'router', 'switch', 'cable', 'internet'],
        'issue': ['issue', 'problem', 'error', 'bug', 'ticket', 'help', 'fix', 'broken']
    }
    
    # Count matches for each category
    matches = {}
    for category, keywords in categories.items():
        matches[category] = sum(1 for keyword in keywords if keyword in query_lower)
    
    # Find the best match
    best_category = max(matches, key=matches.get) if max(matches.values()) > 0 else 'general'
    confidence = max(matches.values())
    
    # Detect if user wants comprehensive information
    comprehensive_keywords = ['all', 'everything', 'complete', 'full', 'entire', 'list', 'show me']
    is_comprehensive = any(keyword in query_lower for keyword in comprehensive_keywords)
    
    return {
        'category': best_category,
        'confidence': confidence,
        'is_comprehensive': is_comprehensive,
        'is_question': any(word in query_lower for word in ['what', 'how', 'where', 'when', 'why', 'which'])
    }

# --- Enhanced PDF Processing ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz) with fallback to PyPDF2"""
    text_content = ""
    
    try:
        # Try PyMuPDF first (better text extraction)
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.get_text()
        doc.close()
        return text_content
    except Exception as e:
        st.warning(f"PyMuPDF failed for {file_path}, trying PyPDF2...")
        
        # Fallback to PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page.extract_text()
            return text_content
        except Exception as e2:
            st.error(f"Both PDF extraction methods failed for {file_path}: {str(e2)}")
            return f"Error extracting PDF content: {str(e2)}"

# --- Enhanced Document Processing ---
@st.cache_resource
def load_documents_enhanced():
    """Load documents with enhanced processing and better user feedback"""
    st.info("🔍 **Loading and processing your documents...**")
    
    documents = []
    file_paths = []
    file_metadata = []
    
    # Enhanced file patterns including PDF
    file_patterns = ["**/*.txt", "**/*.md", "**/*.csv", "**/*.pdf"]
    
    total_files_found = 0
    successful_loads = 0
    
    for pattern in file_patterns:
        files = glob.glob(pattern, recursive=True)
        total_files_found += len(files)
        
        for file_path in files:
            try:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                
                st.info(f"📂 Processing: **{file_name}** ({file_size:,} bytes)")
                
                content = ""
                file_type = ""
                
                if file_path.endswith('.pdf'):
                    # Handle PDF files
                    content = extract_text_from_pdf(file_path)
                    file_type = 'pdf'
                    
                elif file_path.endswith('.csv'):
                    # Enhanced CSV handling
                    try:
                        # Try different encodings
                        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                        df = None
                        
                        for encoding in encodings:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding)
                                break
                            except:
                                continue
                        
                        if df is not None:
                            # Create structured content
                            content = f"📊 **Equipment Database: {file_name}**\n\n"
                            content += f"📈 **Summary:** {len(df)} total items | {len(df.columns)} data fields\n\n"
                            content += f"📋 **Available Fields:** {', '.join(df.columns)}\n\n"
                            
                            # Add searchable equipment data
                            content += "🔍 **Equipment Details:**\n\n"
                            for idx, row in df.head(100).iterrows():  # First 100 items
                                item_details = []
                                for col, val in row.items():
                                    if pd.notna(val) and str(val).strip():
                                        item_details.append(f"{col}: {val}")
                                
                                if item_details:
                                    content += f"**Item {idx + 1}:** {' | '.join(item_details)}\n"
                            
                            file_type = 'csv'
                        else:
                            raise Exception("Could not read CSV with any encoding")
                        
                    except Exception as e:
                        st.warning(f"⚠️ CSV parsing failed, reading as text: {str(e)}")
                        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            content = f"📄 **Data from {file_name}:**\n\n" + f.read()
                        file_type = 'text'
                        
                else:
                    # Handle text files (md, txt)
                    encodings = ['utf-8', 'latin-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except:
                            continue
                    
                    file_type = 'text'
                
                # Validate content
                if content and len(content.strip()) > 50:
                    clean_content = clean_text(content)
                    documents.append(clean_content)
                    file_paths.append(file_path)
                    file_metadata.append({
                        'name': file_name,
                        'path': file_path,
                        'type': file_type,
                        'size': len(clean_content),
                        'original_size': file_size
                    })
                    successful_loads += 1
                    st.success(f"✅ **Successfully loaded:** {file_name}")
                else:
                    st.warning(f"⚠️ **Skipped:** {file_name} (insufficient content)")
                
            except Exception as e:
                st.error(f"❌ **Failed to load** {file_path}: {str(e)}")
    
    # Summary
    if successful_loads > 0:
        st.success(f"🎉 **Successfully loaded {successful_loads}/{total_files_found} documents!**")
    else:
        st.error("❌ **No documents were loaded successfully**")
    
    return documents, file_paths, file_metadata

@st.cache_resource
def create_enhanced_search_index(_documents, _file_paths, _metadata):
    """Create enhanced search index with better chunking and progress tracking"""
    if not _documents:
        return None, None, None, None, None
    
    st.info("🔨 **Building intelligent search index...**")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_chunks = []
    chunk_sources = []
    chunk_metadata = []
    
    # Process each document
    total_docs = len(_documents)
    for i, (doc, file_path, metadata) in enumerate(zip(_documents, _file_paths, _metadata)):
        status_text.text(f"Processing {metadata['name']}...")
        progress_bar.progress((i + 1) / total_docs * 0.5)  # First half of progress
        
        chunks = smart_chunking(doc, config.chunk_size)
        
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_sources.append(file_path)
            chunk_metadata.append(metadata)
    
    st.success(f"📄 **Created {len(all_chunks)} intelligent search chunks**")
    
    # Create embeddings with progress tracking
    status_text.text("🧠 Generating AI embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch_chunks, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        
        # Update progress
        progress = 0.5 + (i / len(all_chunks)) * 0.5
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"🧠 Processing embeddings... {i + len(batch_chunks)}/{len(all_chunks)}")
    
    embeddings = np.vstack(all_embeddings)
    
    # Create optimized FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    progress_bar.progress(1.0)
    status_text.empty()
    progress_bar.empty()
    
    st.success("✅ **Smart search system ready!**")
    
    return index, all_chunks, chunk_sources, chunk_metadata, model

def enhanced_search(index, chunks, chunk_sources, chunk_metadata, model, query: str, k: int = 6):
    """Enhanced search with better relevance and filtering"""
    if index is None or not chunks:
        return []
    
    try:
        query_info = detect_query_intent(query)
        
        # Create query embedding
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search with more results initially for filtering
        search_k = min(k * 2, len(chunks))
        scores, indices = index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        seen_content = set()
        
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and scores[0][i] > config.similarity_threshold:
                content = chunks[idx]
                
                # Avoid near-duplicate content
                content_key = content[:100].lower()
                if content_key not in seen_content:
                    results.append({
                        'content': content,
                        'source': chunk_sources[idx],
                        'metadata': chunk_metadata[idx],
                        'score': float(scores[0][i]),
                        'rank': i + 1,
                        'query_info': query_info
                    })
                    seen_content.add(content_key)
                    
                    if len(results) >= k:
                        break
        
        return results
        
    except Exception as e:
        st.error(f"🔍 Search error: {str(e)}")
        return []

# --- Enhanced Response Generation ---
def format_search_result(content: str, metadata: Dict, query_info: Dict) -> str:
    """Format individual search results for better readability"""
    file_name = metadata.get('name', 'Unknown')
    file_type = metadata.get('type', 'text')
    
    # Determine appropriate emoji based on file type
    emoji = {
        'pdf': '📄',
        'csv': '📊',
        'text': '📝',
        'md': '📋'
    }.get(file_type, '📄')
    
    formatted = f"{emoji} **{file_name}**\n\n"
    
    # For CSV data, extract and format items nicely
    if file_type == 'csv' and 'Item' in content:
        items = re.findall(r'\*\*Item \d+:\*\* (.+)', content)
        if not items:
            items = re.findall(r'Item \d+: (.+)', content)
        
        if items:
            formatted += "🔍 **Found Equipment:**\n\n"
            for i, item in enumerate(items[:4]):  # Show max 4 items
                # Clean up the item formatting
                clean_item = item.replace(' | ', '\n   • ').strip()
                formatted += f"**{i+1}.** {clean_item}\n\n"
        else:
            # Fallback to showing content preview
            preview = content[:300] + "..." if len(content) > 300 else content
            formatted += f"{preview}\n\n"
    else:
        # For other content types, show clean preview
        # Remove markdown formatting for cleaner display
        clean_content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
        clean_content = re.sub(r'\n+', '\n', clean_content)
        
        preview = clean_content[:400] + "..." if len(clean_content) > 400 else clean_content
        formatted += f"{preview}\n\n"
    
    return formatted

def generate_enhanced_response(query: str, search_results: List[Dict]) -> str:
    """Generate user-friendly response with better formatting"""
    query_info = detect_query_intent(query)
    
    if not search_results:
        return f"""🤔 **I couldn't find specific information about "{query}" in your documents.**

💡 **Here's what I can help you with:**
• 📷 Equipment searches (cameras, audio, computers)
• 📊 Inventory lookups
• 📄 Document content searches  
• 🎯 Specific item details

🔍 **Try asking:**
• "Show me all cameras"
• "What audio equipment is available?"
• "List computer equipment"
• "Find projectors"

📚 **Your available documents:**
{chr(10).join([f"• {meta.get('name', 'Unknown')}" for meta in file_metadata]) if 'file_metadata' in globals() else 'Loading...'}"""

    # Start with a friendly header
    response = f"🎯 **Found {len(search_results)} relevant results for: \"{query}\"**\n\n"
    
    # Add category insight if detected
    if query_info['category'] != 'general':
        category_emojis = {
            'camera': '📷',
            'audio': '🎵',
            'computer': '💻',
            'display': '🖥️',
            'network': '🌐',
            'issue': '🔧'
        }
        emoji = category_emojis.get(query_info['category'], '🔍')
        response += f"{emoji} **Category:** {query_info['category'].title()} Equipment\n\n"
    
    response += "---\n\n"
    
    # Group results by source for better organization
    results_by_source = {}
    for result in search_results:
        source_name = os.path.basename(result['source'])
        if source_name not in results_by_source:
            results_by_source[source_name] = []
        results_by_source[source_name].append(result)
    
    # Display results by source
    for source_name, source_results in results_by_source.items():
        if len(source_results) == 1:
            # Single result from this source
            result = source_results[0]
            response += format_search_result(result['content'], result['metadata'], query_info)
        else:
            # Multiple results from same source
            response += f"📁 **Multiple matches in {source_name}:**\n\n"
            for i, result in enumerate(source_results[:3], 1):  # Max 3 per source
                preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                response += f"**Match {i}:** {preview}\n\n"
        
        response += "---\n\n"
    
    # Add helpful suggestions
    response += "💡 **Need something else?**\n\n"
    
    if query_info['category'] == 'camera':
        response += "🔍 Try: *'video recording equipment'* or *'camera accessories'*\n"
    elif query_info['category'] == 'audio':
        response += "🔍 Try: *'microphones and speakers'* or *'sound system'*\n"
    elif query_info['is_comprehensive']:
        response += "🔍 Try: *'cameras'*, *'audio equipment'*, or *'computers'* for specific categories\n"
    else:
        response += "🔍 Try: *'show me all [category]'* or be more specific with model names\n"
    
    return response

# --- Main Streamlit App ---
st.set_page_config(
    page_title="🤖 Smart Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Smart Support Assistant</h1>
    <p>Your intelligent document search companion - now with PDF support!</p>
</div>
""", unsafe_allow_html=True)

# Load data with enhanced feedback
with st.spinner("🚀 **Initializing your smart assistant...**"):
    documents, file_paths, file_metadata = load_documents_enhanced()
    
    if not documents:
        st.error("❌ **No documents found!**")
        st.info("""
        📁 **Please add files to your repository:**
        
        **Supported formats:**
        • 📄 PDF files (.pdf) - *NEW!*
        • 📊 CSV files (.csv) - Equipment lists, inventory
        • 📝 Text files (.txt, .md) - Documentation, manuals
        
        **Instructions:**
        1. Place files in the same directory as this script
        2. Refresh the page
        3. Start chatting!
        """)
        st.stop()

# Create enhanced search index
with st.spinner("🧠 **Building your intelligent search system...**"):
    index, chunks, chunk_sources, chunk_metadata, model = create_enhanced_search_index(
        documents, file_paths, file_metadata
    )
    
    if index is None:
        st.error("❌ **Failed to create search system**")
        st.stop()

# Success message
st.markdown("""
<div class="success-box">
    <h3>🎉 Your Smart Assistant is Ready!</h3>
    <p>I've successfully loaded and processed all your documents. You can now ask me anything!</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.header("📊 **Document Library**")
    
    total_size = sum(meta['original_size'] for meta in file_metadata)
    st.metric("📚 Total Documents", len(file_metadata))
    st.metric("📄 Searchable Chunks", len(chunks) if 'chunks' in locals() else 0)
    st.metric("💾 Total Data Size", f"{total_size:,} bytes")
    
    st.header("📁 **Loaded Files**")
    for meta in file_metadata:
        file_type_emoji = {
            'pdf': '📄',
            'csv': '📊', 
            'text': '📝'
        }.get(meta['type'], '📄')
        
        st.write(f"{file_type_emoji} **{meta['name']}**")
        st.caption(f"Type: {meta['type'].upper()} | Size: {meta['original_size']:,} bytes")
    
    st.header("⚙️ **Search Settings**")
    st.write(f"🎯 Results per query: {config.top_k_retrieval}")
    st.write(f"📏 Similarity threshold: {config.similarity_threshold}")
    st.write(f"🔍 Chunk size: {config.chunk_size} words")
    
    # Quick actions
    st.header("⚡ **Quick Queries**")
    if st.button("📷 Show all cameras"):
        st.session_state.quick_query = "show me all cameras"
    if st.button("🎵 Audio equipment"):
        st.session_state.quick_query = "list audio equipment"  
    if st.button("💻 Computer equipment"):
        st.session_state.quick_query = "show computer equipment"
    if st.button("📋 Everything"):
        st.session_state.quick_query = "show me everything in the inventory"

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Enhanced welcome message
    welcome = f"""👋 **Hello! I'm your Smart Support Assistant!**

🎉 I've successfully loaded **{len(documents)} document(s)** and I'm ready to help you find anything!

**🌟 What makes me special:**
• 🧠 **Intelligent Search** - I understand context and intent
• 📄 **PDF Support** - I can read your PDF documents  
• 📊 **Smart Formatting** - Results are clean and easy to read
• 🎯 **Category Detection** - I automatically categorize your queries

**📁 Your Document Library:**
{chr(10).join([f"• {meta['type'].upper()}: **{meta['name']}**" for meta in file_metadata])}

**🚀 Try asking me:**
• *"What cameras do we have available?"*
• *"Show me audio equipment with wireless capabilities"*
• *"List all computer equipment"*
• *"Find projectors for meeting rooms"*
• *"What's our complete inventory?"*

**💬 What would you like to know?**"""
    
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Handle quick queries
if "quick_query" in st.session_state:
    quick_query = st.session_state.quick_query
    del st.session_state.quick_query
    
    st.session_state.messages.append({"role": "user", "content": quick_query})
    
    # Generate response for quick query
    search_results = enhanced_search(
        index, chunks, chunk_sources, chunk_metadata, model, quick_query, config.top_k_retrieval
    )
    response = generate_enhanced_response(quick_query, search_results)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.experimental_rerun()

# Display chat history with better formatting
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Enhanced chat input
if prompt := st.chat_input("🔍 Ask me about your equipment, documents, or anything else..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate enhanced response
    with st.chat_message("assistant"):
        with st.spinner("🔍 **Searching through your documents with AI...**"):
            # Search for relevant information
            search_results = enhanced_search(
                index, chunks, chunk_sources, chunk_metadata, model, prompt, config.top_k_retrieval
            )
            
            # Generate enhanced response
            response = generate_enhanced_response(prompt, search_results)
            st.markdown(response)
            
            # Optional: Show search details in expander
            if search_results and st.checkbox("🔍 Show search details", key=f"details_{len(st.session_state.messages)}"):
                with st.expander("🔍 **Search Analysis**", expanded=False):
                    query_info = detect_query_intent(prompt)
                    
                    st.write(f"**🎯 Query:** {prompt}")
                    st.write(f"**📂 Category:** {query_info['category'].title()}")
                    st.write(f"**🔍 Results found:** {len(search_results)}")
                    st.write(f"**📊 Confidence:** {query_info['confidence']}/10")
                    
                    st.write("**📋 Top Results:**")
                    for i, result in enumerate(search_results[:3], 1):
                        st.write(f"**{i}.** Score: {result['score']:.3f} | Source: {os.path.basename(result['source'])}")
                        st.caption(f"Preview: {result['content'][:100]}...")
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
