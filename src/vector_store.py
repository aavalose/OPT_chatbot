from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import streamlit as st

class VectorStore:
    # Static cache for the model
    _model = None
    _questions = None
    _embeddings = None
    
    def __init__(self):
        if VectorStore._model is None:
            VectorStore._model = SentenceTransformer('all-MiniLM-L6-v2')
        if VectorStore._questions is None:
            VectorStore._questions = self._load_knowledge_base()
        if VectorStore._embeddings is None:
            VectorStore._embeddings = self._create_embeddings()
        
        self.model = VectorStore._model
        self.questions = VectorStore._questions
        self.embeddings = VectorStore._embeddings
    
    def _load_knowledge_base(self):
        kb_path = Path(__file__).parent.parent / 'data' / 'knowledge_base.json'
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                data = json.load(f)
                return data.get('questions', [])
        return []
    
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