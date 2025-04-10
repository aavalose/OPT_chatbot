from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import streamlit as st

def initialize_vector_store():
    """Initialize the vector store components in session state"""
    if 'model' not in st.session_state:
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if 'questions' not in st.session_state:
        kb_path = Path(__file__).parent.parent / 'data' / 'knowledge_base.json'
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                data = json.load(f)
                st.session_state.questions = data.get('questions', [])
        else:
            st.session_state.questions = []
    
    if 'embeddings' not in st.session_state:
        if st.session_state.questions:
            texts = [q['question'] for q in st.session_state.questions]
            st.session_state.embeddings = st.session_state.model.encode(texts)
        else:
            st.session_state.embeddings = None

def search_similar_questions(query: str, k: int = 1) -> List[Dict]:
    """Search for similar questions using the vector store"""
    # Ensure vector store is initialized
    initialize_vector_store()
    
    if not st.session_state.questions or st.session_state.embeddings is None:
        return []
    
    # Normalize query
    normalized_query = query.lower().replace("opt", "optional practical training (opt)")
    
    # Get query embedding
    query_embedding = st.session_state.model.encode([normalized_query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, st.session_state.embeddings)[0]
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Format results
    results = []
    for idx in top_k_indices:
        question_data = st.session_state.questions[idx].copy()
        question_data['similarity_score'] = float(similarities[idx])
        results.append(question_data)
        
    return results 