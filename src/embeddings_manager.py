import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
import os
from typing import Dict, List
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st


class EmbeddingsManager:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize paths
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.json_dir = self.data_dir / 'json_files'

        # Use Streamlit's cache to initialize ChromaDB only once per session
        self.collection = self.initialize_collection()

    @st.cache_resource
    def initialize_collection():
        """Initialize ChromaDB collection with caching"""
        # Use in-memory client instead of persistent
        chroma_client = chromadb.Client()
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = chroma_client.create_collection(
            name="questions",
            embedding_function=embedding_function
        )
        
        # Load and process knowledge base
        kb_path = Path(__file__).parent.parent / 'data' / 'knowledge_base.json'
        if kb_path.exists():
            try:
                with open(kb_path, 'r') as f:
                    data = json.load(f)
                
                questions = data.get('questions', [])
                
                # Process questions in batches
                batch_size = 32
                for i in range(0, len(questions), batch_size):
                    batch = questions[i:i + batch_size]
                    
                    docs = []
                    metadatas = []
                    ids = []
                    
                    for idx, question_data in enumerate(batch, start=i):
                        question = question_data.get('question', '')
                        answer = question_data.get('answer', '')
                        category = question_data.get('category', '')
                        metadata = question_data.get('metadata', {})
                        
                        # Create augmented question text that includes both forms
                        augmented_question = question
                        if "OPT" in question and "Optional Practical Training" not in question:
                            augmented_question = question.replace("OPT", "Optional Practical Training (OPT)")
                        elif "Optional Practical Training" in question and "(OPT)" not in question:
                            augmented_question = question.replace("Optional Practical Training", "Optional Practical Training (OPT)")
                        
                        docs.append(augmented_question)
                        metadatas.append({
                            'original_question': question,
                            'answer': answer,
                            'category': category,
                            'metadata': json.dumps(metadata),
                        })
                        ids.append(f"q_{idx}")
                    
                    # Add batch to ChromaDB collection
                    collection.add(
                        documents=docs,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                print(f"Processed knowledge base with {len(questions)} questions")
                
            except Exception as e:
                print(f"Error processing knowledge base: {str(e)}")
                raise e
        
        return collection

    def search_similar_questions(self, query: str, k: int = 1, print_results: bool = False) -> List[Dict]:
        """
        Search for similar questions in the vector store
        
        Args:
            query (str): The question to search for
            k (int): Number of similar questions to return
            
        Returns:
            List[Dict]: List of similar questions with their metadata
        """
        try:
            # Normalize the query to handle common variations
            normalized_query = query.lower().replace("opt", "optional practical training (opt)")
            
            results = self.collection.query(
                query_texts=[normalized_query],
                n_results=k
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = results['metadatas'][0][i].copy()
                result['question'] = results['documents'][0][i]
                distance = float(results['distances'][0][i])
                # Adjust similarity calculation to be more discriminative
                result['similarity_score'] = 1.0 / (1.0 + distance)
                formatted_results.append(result)
            
            # Sort by similarity score in descending order
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Add debug information
            if print_results:
                print(f"Query: {query}")
                print(f"Normalized query: {normalized_query}")
                print(f"Raw distances: {results['distances'][0]}")
                print(f"Converted scores: {[r['similarity_score'] for r in formatted_results]}")
                
            return formatted_results
            
        except Exception as e:
            print(f"Error searching similar questions: {str(e)}")
            return []

    def reset_vector_store(self):
        """Reset the vector store by deleting and recreating the collection"""
        try:
            # Delete the existing collection
            self.collection.delete()
            print("Deleted existing collection")
            
            # Create a new collection
            self.collection = self.initialize_collection()
            print("Created new empty collection")
            
            return True
        except Exception as e:
            print(f"Error resetting vector store: {str(e)}")
            return False 