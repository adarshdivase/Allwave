# pages/2_AI_Support_Chatbot.py
import streamlit as st
import os
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub # To simulate a real LLM

# --- Backend Functions ---
@st.cache_resource
def create_knowledge_base():
    """
    Loads all documents, splits them, creates embeddings, and stores them in a FAISS vector store.
    This function runs only once and is cached.
    """
    # Load all supported file types from the current directory
    loader = DirectoryLoader('.', glob="**/*.*", loader_cls=UnstructuredFileLoader, use_multithreading=True, show_progress=True)
    documents = loader.load()

    if not documents:
        st.error("No documents found to load. Please add your data files to the project folder.")
        return None

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store and save it locally
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    return db

def run_query(db, query):
    """
    Searches the knowledge base for relevant documents for a given query.
    For this demo, we will show the retrieved documents instead of generating a summary.
    """
    if db is None:
        return []
    
    # Retrieve the most relevant document chunks
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retrieved_docs = retriever.get_relevant_documents(query)
    return retrieved_docs

# --- Streamlit App ---
st.title("AI Support Chatbot 🧠")
st.write("Ask any question about past projects, technical issues, or meeting notes.")

# Create/load the knowledge base
with st.spinner("Preparing AI Assistant... This may take a moment on first run."):
    try:
        db = create_knowledge_base()
        if db:
            st.success("AI Assistant is ready.")
        else:
            st.stop()
    except Exception as e:
        st.error(f"Failed to initialize the knowledge base: {e}")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., What were the audio issues at the BCG Mumbai site?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response_docs = run_query(db, prompt)
            
            if not response_docs:
                response = "I couldn't find any relevant information in the knowledge base."
            else:
                # In a full app, you'd feed these docs to an LLM. Here, we display the sources.
                response = "I found the following relevant information from your documents:\n\n"
                for i, doc in enumerate(response_docs):
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    response += f"**Source {i+1} ({source_name}):**\n"
                    response += f"> {doc.page_content[:300]}...\n\n"

            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
