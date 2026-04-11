# frontend_streamlit/model.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_resource
def load_nlp_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_index(_model, text_data):
    embeddings = _model.encode(text_data, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index