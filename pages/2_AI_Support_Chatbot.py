# pages/2_AI_Support_Chatbot.py
import streamlit as st
import os
import glob
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore")

# --- Backend Functions ---
@st.cache_resource
def create_knowledge_base():
    """
    Loads documents, splits them, creates embeddings, and stores them in a FAISS vector store.
    This simplified version only handles txt, csv, and md files to avoid dependency issues.
    """
    try:
        documents = []
        supported_files = []
        
        # Find all supported files
        for ext in ['*.txt', '*.csv', '*.md']:
            files = glob.glob(f"**/{ext}", recursive=True)
            supported_files.extend(files)
        
        if not supported_files:
            st.error("No supported files found. Please add .txt, .csv, or .md files to your project.")
            st.write("Supported formats: Text files (.txt), CSV files (.csv), Markdown files (.md)")
            return None
        
        # Load each file
        for file_path in supported_files:
            try:
                if file_path.endswith('.csv'):
                    loader = CSVLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                docs = loader.load()
                documents.extend(docs)
                st.success(f"✅ Loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load {file_path}: {str(e)}")
        
        if not documents:
            st.error("No documents could be loaded successfully.")
            return None
        
        st.info(f"📚 Successfully loaded {len(documents)} documents")
        
        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        
        st.info(f"📄 Created {len(docs)} text chunks for search")
        
        # Create embeddings with CPU-only specification
        embeddings = SentenceTransformerEmbeddings(
            model_name="paraphrase-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create FAISS vector store
        db = FAISS.from_documents(docs, embeddings)
        
        # Save locally for faster subsequent loads
        try:
            db.save_local("faiss_index")
        except:
            pass  # Continue even if save fails
        
        return db
        
    except Exception as e:
        st.error(f"❌ Error creating knowledge base: {str(e)}")
        st.write("**Try these solutions:**")
        st.write("1. Ensure you have .txt, .csv, or .md files in your repository")
        st.write("2. Check that your requirements.txt has the correct dependencies")
        st.write("3. Use 'Reboot app' from the menu if the issue persists")
        return None

@st.cache_resource
def load_existing_db():
    """Try to load existing FAISS database if available"""
    try:
        if os.path.exists("faiss_index"):
            embeddings = SentenceTransformerEmbeddings(
                model_name="paraphrase-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            db = FAISS.load_local("faiss_index", embeddings)
            return db
    except:
        pass
    return None

def run_query(db, query):
    """
    Searches the knowledge base for relevant documents for a given query.
    """
    if db is None:
        return []
    
    try:
        # Retrieve the most relevant document chunks
        retriever = db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        retrieved_docs = retriever.get_relevant_documents(query)
        return retrieved_docs
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return []

# --- Streamlit App ---
st.set_page_config(page_title="AI Support Chatbot", page_icon="🧠")

st.title("AI Support Chatbot 🧠")
st.write("Ask any question about your documents and projects.")

# Set environment variables for better compatibility
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

# Try to load existing database first, then create new one if needed
db = None

with st.spinner("🔄 Initializing AI Assistant..."):
    # First try to load existing database
    db = load_existing_db()
    
    if db is None:
        # Create new knowledge base
        db = create_knowledge_base()
    else:
        st.success("✅ AI Assistant loaded from cache")

if db is None:
    st.error("❌ Could not initialize the AI Assistant")
    st.write("Please check your files and requirements, then use 'Reboot app' from the menu.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I'm your AI Support Assistant. I can help you find information from your uploaded documents. What would you like to know?"
    })

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching through your documents..."):
            response_docs = run_query(db, prompt)
            
            if not response_docs:
                response = "I couldn't find any relevant information in your documents for that query. Try rephrasing your question or check if the relevant documents are uploaded."
            else:
                response = "Here's what I found in your documents:\n\n"
                
                for i, doc in enumerate(response_docs, 1):
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    content = doc.page_content.strip()
                    
                    # Limit content length for better readability
                    if len(content) > 400:
                        content = content[:400] + "..."
                    
                    response += f"**📄 Source {i}: {source_file}**\n"
                    response += f"{content}\n\n"
                    response += "---\n\n"
                
                response += "*💡 Tip: You can ask follow-up questions to get more specific information.*"
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with app info
with st.sidebar:
    st.header("📋 App Info")
    st.write("This chatbot searches through your uploaded documents to answer questions.")
    
    st.header("📁 Supported Files")
    st.write("- Text files (.txt)")
    st.write("- CSV files (.csv)")  
    st.write("- Markdown files (.md)")
    
    if st.button("🔄 Refresh Knowledge Base"):
        st.cache_resource.clear()
        st.rerun()
