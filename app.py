# app.py
import streamlit as st

st.set_page_config(
    page_title="All Wave AI Platform",
    layout="wide"
)

st.title("Welcome to the All Wave AI Design & Support Platform 🚀")
st.sidebar.success("Select a tool above.")

st.markdown(
    """
    This is the central hub for our next-generation AV solutions platform.
    
    **👈 Select a tool from the sidebar** to get started.

    ### Tools Available:
    - **Interactive Configurator:** A tool for customers to design their own meeting rooms and get instant visual feedback and equipment recommendations.
    - **AI Support Chatbot:** An intelligent assistant trained on our entire history of support tickets, project documents, and meeting notes to help diagnose issues and answer questions.
    """
)
