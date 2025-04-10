from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import streamlit as st

def initialize_vector_store():
    """Initialize the vector store components in session state"""
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    if 'questions' not in st.session_state:
        kb_path = Path(__file__).parent.parent / 'data' / 'knowledge_base.json'
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                data = json.load(f)
                st.session_state.questions = data.get('questions', [])
        else:
            st.session_state.questions = []
    
    if 'embeddings' not in st.session_state and st.session_state.questions:
        texts = [q['question'] for q in st.session_state.questions]
        st.session_state.embeddings = st.session_state.vectorizer.fit_transform(texts)

def search_similar_questions(query: str, k: int = 1) -> List[Dict]:
    """Search for similar questions using TF-IDF and cosine similarity"""
    # Ensure vector store is initialized
    initialize_vector_store()
    
    if not st.session_state.questions:
        return []
    
    # Normalize query
    normalized_query = query.lower().replace("opt", "optional practical training (opt)")
    
    # Get query embedding
    query_vector = st.session_state.vectorizer.transform([normalized_query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, st.session_state.embeddings)[0]
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Format results
    results = []
    for idx in top_k_indices:
        question_data = st.session_state.questions[idx].copy()
        question_data['similarity_score'] = float(similarities[idx])
        results.append(question_data)
        
    return results 