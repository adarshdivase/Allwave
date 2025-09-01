# pages/2_AI_Support_Chatbot.py
import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

# --- Backend Functions ---
@st.cache_resource
def create_knowledge_base():
    """
    Loads all documents, splits them, creates embeddings, and stores them in a FAISS vector store.
    """
    loader = DirectoryLoader('.', glob="**/*.*", loader_cls=UnstructuredFileLoader, use_multithreading=True, show_progress=True)
    documents = loader.load()

    if not documents:
        st.error("No documents found to load. Please add your data files to the project folder.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    return db

def run_query(db, query):
    """
    Searches the knowledge base for relevant documents for a given query.
    """
    if db is None: return []
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
    retrieved_docs = retriever.get_relevant_documents(query)
    return retrieved_docs

# --- Streamlit App ---
st.title("AI Support Chatbot 🧠")
st.write("Ask any question about past projects, technical issues, or meeting notes.")

with st.spinner("Preparing AI Assistant... This may take a moment on first run."):
    try:
        db = create_knowledge_base()
        if db: st.success("AI Assistant is ready.")
        else: st.stop()
    except Exception as e:
        st.error(f"Failed to initialize the knowledge base: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What were the audio issues at the BCG Mumbai site?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            response_docs = run_query(db, prompt)
            
            if not response_docs:
                response = "I couldn't find any relevant information in the knowledge base."
            else:
                response = "I found the following relevant information from your documents:\n\n"
                for i, doc in enumerate(response_docs):
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    response += f"**Source {i+1} ({source_name}):**\n"
                    response += f"> {doc.page_content[:300]}...\n\n"
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
