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
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# --- Enhanced RAG Configuration ---
class RAGConfig:
    def __init__(self):
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.top_k_retrieval = 5
        self.similarity_threshold = 0.3
        self.max_context_length = 2000
        self.use_reranking = True

config = RAGConfig()

# --- Advanced Text Processing ---
def advanced_text_preprocessing(text: str) -> str:
    """Advanced text preprocessing with domain-specific cleaning"""
    # Remove special characters but keep meaningful punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Handle common IT/JIRA terminology
    text = re.sub(r'\b(ID|id):\s*(\w+)', r'identifier \2', text)
    text = re.sub(r'\b(P\d+|HIGH|LOW|MEDIUM)\b', lambda m: f'priority_{m.group().lower()}', text)
    return text

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract key entities from text using simple pattern matching"""
    entities = {
        'equipment': [],
        'issues': [],
        'priorities': [],
        'dates': [],
        'locations': []
    }
    
    # Equipment patterns
    equipment_patterns = [
        r'\b(camera|laptop|pc|computer|microphone|speaker|projector|monitor)\w*\b',
        r'\b(audio|video|hardware|software)\s+\w+\b'
    ]
    
    for pattern in equipment_patterns:
        entities['equipment'].extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Issue patterns
    issue_patterns = [
        r'\b(error|bug|issue|problem|failure|crash)\w*\b',
        r'\b(not working|broken|failed|timeout)\b'
    ]
    
    for pattern in issue_patterns:
        entities['issues'].extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Priority patterns
    entities['priorities'] = re.findall(r'\b(high|medium|low|critical|urgent)\b', text, re.IGNORECASE)
    
    # Date patterns
    entities['dates'] = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', text)
    
    # Location patterns
    entities['locations'] = re.findall(r'\b(mumbai|delhi|bangalore|chennai|office|room\s*\d+)\b', text, re.IGNORECASE)
    
    return entities

# --- Intelligent Chunking ---
def intelligent_chunking(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """Create intelligent chunks with metadata"""
    chunks = []
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            entities = extract_entities(chunk_text)
            
            chunks.append({
                'text': chunk_text,
                'entities': entities,
                'size': current_size,
                'sentence_count': len(current_chunk)
            })
            
            # Handle overlap
            if overlap > 0 and len(current_chunk) > 1:
                current_chunk = current_chunk[-1:]
                current_size = len(current_chunk[0])
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        entities = extract_entities(chunk_text)
        chunks.append({
            'text': chunk_text,
            'entities': entities,
            'size': current_size,
            'sentence_count': len(current_chunk)
        })
    
    return chunks

# --- Query Understanding ---
def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze query to understand user intent"""
    query_lower = query.lower()
    
    intent_patterns = {
        'equipment_inquiry': [
            r'\b(what|show|list|find).*\b(camera|laptop|pc|audio|video|equipment)\b',
            r'\b(which|what type).*\b(device|hardware|equipment)\b'
        ],
        'issue_tracking': [
            r'\b(issue|problem|error|bug|ticket)\b',
            r'\b(not working|broken|failed|down)\b',
            r'\b(report|status|update).*\b(issue|problem)\b'
        ],
        'project_status': [
            r'\b(project|status|progress|update)\b',
            r'\b(how.*going|what.*happening)\b'
        ],
        'comparison': [
            r'\b(compare|difference|better|vs|versus)\b',
            r'\b(which.*better|should.*choose)\b'
        ],
        'recommendation': [
            r'\b(recommend|suggest|advice|best)\b',
            r'\b(what.*should|which.*use)\b'
        ]
    }
    
    detected_intents = []
    for intent, patterns in intent_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                detected_intents.append(intent)
                break
    
    # Extract key entities from query
    query_entities = extract_entities(query)
    
    return {
        'intents': detected_intents,
        'entities': query_entities,
        'query_type': 'factual' if any(word in query_lower for word in ['what', 'when', 'where', 'who', 'how']) else 'action',
        'complexity': 'complex' if len(query.split()) > 10 or '?' in query else 'simple'
    }

# --- Enhanced Retrieval with Reranking ---
def rerank_results(query: str, results: List[Dict], query_analysis: Dict) -> List[Dict]:
    """Rerank results based on query intent and entity matching"""
    def calculate_relevance_score(result, query_analysis):
        score = result['score']  # Base similarity score
        
        # Boost based on entity matching
        result_entities = result.get('entities', {})
        query_entities = query_analysis['entities']
        
        entity_boost = 0
        for entity_type in query_entities:
            if entity_type in result_entities:
                common_entities = set(query_entities[entity_type]) & set(result_entities[entity_type])
                entity_boost += len(common_entities) * 0.1
        
        # Intent-based boosting
        content_lower = result['content'].lower()
        for intent in query_analysis['intents']:
            if intent == 'equipment_inquiry' and any(word in content_lower for word in ['camera', 'laptop', 'audio', 'video']):
                score += 0.2
            elif intent == 'issue_tracking' and any(word in content_lower for word in ['error', 'issue', 'problem', 'ticket']):
                score += 0.2
            elif intent == 'project_status' and any(word in content_lower for word in ['project', 'status', 'progress']):
                score += 0.2
        
        return score + entity_boost
    
    # Calculate enhanced scores
    for result in results:
        result['enhanced_score'] = calculate_relevance_score(result, query_analysis)
    
    # Sort by enhanced score
    return sorted(results, key=lambda x: x['enhanced_score'], reverse=True)

# --- Smart Response Generation ---
def generate_smart_response(query: str, results: List[Dict], query_analysis: Dict) -> str:
    """Generate contextually aware responses using retrieved information"""
    
    if not results:
        return generate_no_results_response(query, query_analysis)
    
    # Group results by source type and intent
    grouped_results = group_results_by_relevance(results, query_analysis)
    
    # Generate response based on intent
    primary_intent = query_analysis['intents'][0] if query_analysis['intents'] else 'general'
    
    response_generators = {
        'equipment_inquiry': generate_equipment_response,
        'issue_tracking': generate_issue_response,
        'project_status': generate_project_response,
        'comparison': generate_comparison_response,
        'recommendation': generate_recommendation_response
    }
    
    generator = response_generators.get(primary_intent, generate_general_response)
    return generator(query, grouped_results, query_analysis)

def generate_equipment_response(query: str, grouped_results: Dict, query_analysis: Dict) -> str:
    """Generate equipment-focused response"""
    response = "Based on your equipment inventory and documentation, here's what I found:\n\n"
    
    equipment_items = []
    for result in grouped_results.get('high_relevance', [])[:3]:
        content = result['content']
        # Extract equipment information
        equipment_matches = re.findall(r'\b(camera|laptop|pc|audio|video|microphone|speaker)\w*[^.]*', content, re.IGNORECASE)
        equipment_items.extend(equipment_matches)
    
    if equipment_items:
        response += "**Available Equipment:**\n"
        for i, item in enumerate(set(equipment_items[:5]), 1):
            response += f"{i}. {item.strip()}\n"
        response += "\n"
    
    # Add contextual information
    response += generate_contextual_details(grouped_results)
    
    return response

def generate_issue_response(query: str, grouped_results: Dict, query_analysis: Dict) -> str:
    """Generate issue-focused response"""
    response = "Here's what I found regarding issues and tickets:\n\n"
    
    issues = []
    priorities = []
    
    for result in grouped_results.get('high_relevance', [])[:3]:
        entities = result.get('entities', {})
        issues.extend(entities.get('issues', []))
        priorities.extend(entities.get('priorities', []))
    
    if issues:
        response += "**Identified Issues:**\n"
        for issue in set(issues[:5]):
            response += f"• {issue}\n"
        response += "\n"
    
    if priorities:
        response += f"**Priority Levels:** {', '.join(set(priorities))}\n\n"
    
    response += generate_contextual_details(grouped_results)
    
    return response

def generate_general_response(query: str, grouped_results: Dict, query_analysis: Dict) -> str:
    """Generate general response with smart summarization"""
    response = "Based on your query, here's the most relevant information:\n\n"
    
    # Prioritize high-relevance results
    top_results = grouped_results.get('high_relevance', [])[:2]
    if not top_results:
        top_results = grouped_results.get('medium_relevance', [])[:2]
    
    for i, result in enumerate(top_results, 1):
        source_name = os.path.basename(result['source'])
        content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
        
        response += f"**{i}. From {source_name}:**\n"
        response += f"{content}\n\n"
    
    # Add smart suggestions
    response += generate_smart_suggestions(query, query_analysis)
    
    return response

def generate_contextual_details(grouped_results: Dict) -> str:
    """Generate additional contextual details"""
    details = ""
    
    # Add source information
    sources = set()
    for category in grouped_results.values():
        for result in category:
            sources.add(os.path.basename(result['source']))
    
    if sources:
        details += f"**Sources consulted:** {', '.join(sources)}\n\n"
    
    return details

def generate_smart_suggestions(query: str, query_analysis: Dict) -> str:
    """Generate smart follow-up suggestions"""
    suggestions = "💡 **You might also want to ask:**\n"
    
    if 'equipment' in str(query_analysis['entities']):
        suggestions += "• 'What are the specifications of [equipment name]?'\n"
        suggestions += "• 'Are there any issues with this equipment?'\n"
    
    if query_analysis.get('complexity') == 'simple':
        suggestions += "• Try asking for more specific details\n"
        suggestions += "• Ask for comparisons between options\n"
    
    return suggestions

def generate_no_results_response(query: str, query_analysis: Dict) -> str:
    """Generate helpful response when no results found"""
    response = "I couldn't find specific information about that in your documents, but let me help you:\n\n"
    
    # Provide suggestions based on query analysis
    if 'equipment' in str(query_analysis['entities']):
        response += "**For equipment queries, try:**\n"
        response += "• 'Show me all cameras'\n"
        response += "• 'List audio equipment'\n"
        response += "• 'What laptops are available?'\n\n"
    
    response += "**General tips:**\n"
    response += "• Try using different keywords\n"
    response += "• Ask about specific categories (equipment, issues, projects)\n"
    response += "• Check if your documents contain the information you're looking for\n"
    
    return response

def group_results_by_relevance(results: List[Dict], query_analysis: Dict) -> Dict:
    """Group results by relevance levels"""
    grouped = {
        'high_relevance': [],
        'medium_relevance': [],
        'low_relevance': []
    }
    
    for result in results:
        score = result.get('enhanced_score', result['score'])
        if score > 0.7:
            grouped['high_relevance'].append(result)
        elif score > 0.4:
            grouped['medium_relevance'].append(result)
        else:
            grouped['low_relevance'].append(result)
    
    return grouped

# --- Enhanced Document Processing ---
@st.cache_resource
def load_and_process_documents():
    """Enhanced document loading with better processing"""
    documents = []
    file_paths = []
    document_metadata = []
    
    file_patterns = {
        "**/*.txt": "text",
        "**/*.md": "text", 
        "**/*.csv": "csv",
        "**/*.json": "json"
    }
    
    for pattern, file_type in file_patterns.items():
        files = glob.glob(pattern, recursive=True)
        
        for file_path in files:
            try:
                metadata = {
                    'file_path': file_path,
                    'file_type': file_type,
                    'file_name': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path))
                }
                
                if file_type == "csv":
                    df = pd.read_csv(file_path, encoding='latin-1')
                    content = f"CSV Data from {metadata['file_name']}:\n"
                    content += df.head(10).to_string(index=False)  # Limit to first 10 rows
                    metadata['rows'] = len(df)
                    metadata['columns'] = list(df.columns)
                elif file_type == "json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        content = f"JSON Data from {metadata['file_name']}:\n"
                        content += json.dumps(json_data, indent=2)[:1000]
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                if content.strip():
                    processed_content = advanced_text_preprocessing(content)
                    documents.append(processed_content)
                    file_paths.append(file_path)
                    document_metadata.append(metadata)
                    st.success(f"✅ Loaded: {metadata['file_name']}")
                    
            except Exception as e:
                st.warning(f"⚠️ Skipped {file_path}: {str(e)}")
    
    return documents, file_paths, document_metadata

@st.cache_resource
def create_enhanced_search_index(_documents, _file_paths, _metadata):
    """Create enhanced search index with intelligent chunking"""
    if not _documents:
        return None, None, None, None, None
    
    all_chunks = []
    chunk_sources = []
    chunk_metadata = []
    
    for i, doc in enumerate(_documents):
        chunks = intelligent_chunking(doc, config.chunk_size, config.chunk_overlap)
        
        for chunk in chunks:
            all_chunks.append(chunk['text'])
            chunk_sources.append(_file_paths[i])
            chunk_metadata.append({
                **_metadata[i],
                **chunk
            })
    
    st.info(f"📄 Created {len(all_chunks)} intelligent chunks with metadata")
    
    # Use a more powerful sentence transformer
    model = SentenceTransformer('all-MiniLM-L12-v2')  # Better model
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)
    
    # Create FAISS index with better configuration
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    return index, all_chunks, chunk_sources, chunk_metadata, model

def enhanced_search(index, chunks, chunk_sources, chunk_metadata, model, query: str) -> List[Dict]:
    """Enhanced search with query analysis and reranking"""
    if index is None:
        return []
    
    try:
        # Analyze query intent
        query_analysis = analyze_query_intent(query)
        
        # Enhance query with domain-specific terms
        enhanced_query = enhance_query_with_domain_knowledge(query, query_analysis)
        
        # Perform vector search
        query_embedding = model.encode([enhanced_query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding.astype('float32'), config.top_k_retrieval)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and scores[0][i] > config.similarity_threshold:
                result = {
                    'content': chunks[idx],
                    'source': chunk_sources[idx],
                    'metadata': chunk_metadata[idx],
                    'score': float(scores[0][i]),
                    'entities': chunk_metadata[idx].get('entities', {})
                }
                results.append(result)
        
        # Apply reranking if enabled
        if config.use_reranking and results:
            results = rerank_results(query, results, query_analysis)
        
        return results[:config.top_k_retrieval]
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def enhance_query_with_domain_knowledge(query: str, query_analysis: Dict) -> str:
    """Enhance query with domain-specific knowledge"""
    enhanced = query
    
    # Add synonyms and related terms
    domain_expansions = {
        'camera': 'camera video recording device',
        'laptop': 'laptop computer pc workstation',
        'issue': 'issue problem error bug ticket',
        'audio': 'audio sound microphone speaker'
    }
    
    for term, expansion in domain_expansions.items():
        if term in query.lower():
            enhanced += f" {expansion}"
    
    return enhanced

# --- Main Streamlit Interface ---
st.set_page_config(page_title="Smart AI Support Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Smart AI Support Assistant")
st.markdown("*Powered by Advanced RAG - Intelligent retrieval and contextual understanding*")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ RAG Configuration")
    
    config.top_k_retrieval = st.slider("Results to retrieve", 3, 10, 5)
    config.similarity_threshold = st.slider("Similarity threshold", 0.1, 0.8, 0.3)
    config.use_reranking = st.checkbox("Enable smart reranking", True)
    
    st.header("📊 System Status")

# Initialize the enhanced system
with st.spinner("🔄 Initializing Smart RAG System..."):
    documents, file_paths, metadata = load_and_process_documents()
    
    if not documents:
        st.error("❌ No documents found!")
        st.markdown("""
        **Please add documents to your repository:**
        - Text files (.txt, .md)
        - CSV files with JIRA/ticket data (.csv)
        - JSON configuration files (.json)
        """)
        st.stop()

with st.spinner("🧠 Building Enhanced Search Index..."):
    index, chunks, chunk_sources, chunk_metadata, model = create_enhanced_search_index(
        documents, file_paths, metadata
    )
    
    if index is None:
        st.error("❌ Failed to create search index")
        st.stop()

st.success("✅ Smart RAG System Ready! Ask me anything!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = """Hello! 👋 I'm your Smart AI Support Assistant powered by advanced RAG technology.

I can provide intelligent, contextual answers about your documents and JIRA tickets. I understand intent, extract entities, and provide logical reasoning.

**What makes me smart:**
- 🧠 Intent understanding and entity extraction
- 🔍 Advanced retrieval with reranking
- 📊 Contextual response generation
- 💡 Smart suggestions and follow-ups

**Try these types of questions:**
• "Compare the audio equipment options we have"
• "What critical issues need immediate attention?"
• "Recommend the best camera for video meetings"
• "Show me all network-related problems from last month"

What would you like to know?"""
    
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your smart question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your question and searching intelligently..."):
            # Analyze query
            query_analysis = analyze_query_intent(prompt)
            
            # Perform enhanced search
            search_results = enhanced_search(
                index, chunks, chunk_sources, chunk_metadata, model, prompt
            )
            
            # Generate smart response
            response = generate_smart_response(prompt, search_results, query_analysis)
            st.markdown(response)
            
            # Show debug info in expander
            with st.expander("🔍 Query Analysis & Results"):
                st.json({
                    "detected_intents": query_analysis['intents'],
                    "extracted_entities": query_analysis['entities'],
                    "query_type": query_analysis['query_type'],
                    "results_found": len(search_results),
                    "top_sources": [os.path.basename(r['source']) for r in search_results[:3]]
                })
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Update sidebar stats
with st.sidebar:
    if 'documents' in locals():
        st.metric("📄 Documents", len(documents))
    if 'chunks' in locals():
        st.metric("🔍 Smart Chunks", len(chunks))
    
    st.header("📈 Performance")
    st.metric("🎯 Retrieval Method", "Vector + Rerank")
    st.metric("🧠 Model", "all-MiniLM-L12-v2")
    
    st.header("💡 Smart Features")
    st.markdown("""
    ✅ Intent Detection
    ✅ Entity Extraction  
    ✅ Contextual Reranking
    ✅ Smart Chunking
    ✅ Query Enhancement
    ✅ Logical Reasoning
    """)
    
    if st.button("🔄 Refresh System"):
        st.cache_resource.clear()
        st.rerun()
