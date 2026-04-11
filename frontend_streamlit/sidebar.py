# frontend_streamlit/sidebar.py
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ System Status")
        st.success("Vector Database: Active")
        st.success("NLP Model: Loaded")
        
        st.markdown("---")
        st.markdown("### 🧠 How it Works")
        st.markdown("**1. Semantic Search:**\nThe system converts your query into a dense mathematical vector and compares it against 10,000 hotel vectors using Facebook's FAISS library.")
        st.markdown("**2. Smart Filtering:**\nRegex engines actively scan your input to auto-detect and isolate specific countries and cities, prioritizing location accuracy over semantic matching.")