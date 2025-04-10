from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import streamlit as st

class VectorStore:
    @st.cache_resource
    def _initialize_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    @st.cache_data
    def _load_knowledge_base():
        kb_path = Path(__file__).parent.parent / 'data' / 'knowledge_base.json'
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                data = json.load(f)
                return data.get('questions', [])
        return []
    
    def __init__(self):
        self.model = self._initialize_model()
        self.questions = self._load_knowledge_base()
        self.embeddings = self._create_embeddings()
    
    @st.cache_data
    def _create_embeddings(self):
        if not self.questions:
            return None
        texts = [q['question'] for q in self.questions]
        return self.model.encode(texts)
    
    def search_similar_questions(self, query: str, k: int = 1) -> List[Dict]:
        if not self.questions or self.embeddings is None:
            return []
        
        # Normalize query
        normalized_query = query.lower().replace("opt", "optional practical training (opt)")
        
        # Get query embedding
        query_embedding = self.model.encode([normalized_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Format results
        results = []
        for idx in top_k_indices:
            question_data = self.questions[idx].copy()
            question_data['similarity_score'] = float(similarities[idx])
            results.append(question_data)
            
        return results 