import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict
from .embeddings_manager import *
import streamlit as st

# Try to get API key from different sources
def get_api_key():
    # First try Streamlit secrets (for cloud deployment)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except:
        # If not in Streamlit Cloud, try local .env file
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables or Streamlit secrets")

# Configure the Gemini API with your key
try:
    api_key = get_api_key()
    genai.configure(api_key=api_key)
    print("API Key loaded successfully")
except Exception as e:
    print(f"Error loading API key: {str(e)}")

def preprocess_query(query: str) -> Tuple[List[str], List[Dict]]:
    """
    Enhanced preprocessing that includes semantic search
    
    Args:
        query (str): User's question
        
    Returns:
        Tuple[List[str], List[Dict]]: Categories and similar questions
    """
    model = None
    response = None
    try:
        generation_config = {
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 50,
            'max_output_tokens': 2048,
        }
        
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config
        )
        
        prompt = f"""Analyze this question and return up to THREE most relevant categories from the following list, ordered by relevance:
         - General Information and Eligibility: Questions about what OPT is, who qualifies, F-1 status requirements, program of study requirements, etc.
        - Application Process: Questions about how to apply for OPT, required documents, USCIS forms, application fees, etc.
        - Important Dates: Questions about OPT start/end dates, application deadlines, grace periods, STEM extension timelines, etc.
        - Employment and Unemployment Requirements: Questions about job offers, allowed work types, unemployment limits, employer requirements, etc.
        - Reporting Requirements: Questions about reporting jobs to DSO, address changes, SEVIS updates, employer information, etc.
        - Travel Information: Questions about traveling while on OPT, visa stamps, re-entry procedures, risks of leaving the US, etc.
        - Other: Any OPT-related questions that don't fit into the above categories

        Examples:
        Question: "How do I apply for OPT?"
        Categories: Application Process, General Information and Eligibility

        Question: "When should I submit my OPT application?"
        Categories: Important Dates, Application Process

        Question: "Can I travel outside the US while on OPT?"
        Categories: Travel Information

        Question: "How many days can I be unemployed during OPT?"
        Categories: Employment and Unemployment Requirements, Important Dates

        Question: "Do I need to report my new job to my DSO?"
        Categories: Reporting Requirements, Employment and Unemployment Requirements
        
        Your question: "{query}"
            
            Return only the category names in a comma-separated list, nothing else."""
        
        response = model.generate_content(prompt)
        categories = [cat.strip() for cat in response.text.split(',')]
        
        # Validate categories
        valid_categories = {
            'General Information and Eligibility',
            'Application Process',
            'Important Dates',
            'Employment and Unemployment Requirements',
            'Reporting Requirements',
            'Travel Information',
            'Other'
        }
        
        validated_categories = [cat for cat in categories if cat in valid_categories]
        return (validated_categories if validated_categories else ['Other'])
        
    except Exception as e:
        print(f"Error in preprocessing query: {str(e)}")
        return ['Other']
    finally:
        # Explicit cleanup
        if response:
            del response
        if model:
            del model

def match_files(categories: List[str]) -> Tuple[str, List[str]]:
    """
    Match categories to corresponding JSON files.
    
    Args:
        categories (List[str]): List of categories to match
        
    Returns:
        Tuple[str, List[str]]: Primary document and list of secondary documents
    """    
    # Ensure categories are strings
    categories = [str(cat) if isinstance(cat, list) else cat for cat in categories]
    
    category_to_file = {
        'General Information and Eligibility': 'general_info.json',
        'Application Process': 'application_process.json',
        'Important Dates': 'dates.json',
        'Employment and Unemployment Requirements': 'employment_requirements.json',
        'Reporting Requirements': 'reporting_requirements.json',
        'Travel Information': 'travel.json',
        'Other': None
    }
    
    # Get the data directory path
    data_dir = Path(__file__).parent.parent / 'data' / 'json_files'
    
    # Match and verify files exist
    matched_files = []
    for category in categories:
        if category in category_to_file and category_to_file[category]:
            file_path = data_dir / category_to_file[category]
            if file_path.exists():
                matched_files.append(category_to_file[category])
    
    primary_document = matched_files[0] if matched_files else ""
    secondary_documents = matched_files[1:] if len(matched_files) > 1 else []
    
    return primary_document, secondary_documents

def load_json_content(filename: str) -> dict:
    """
    Load and return content from a JSON file.
    
    Args:
        filename (str): Name of the JSON file to load
        
    Returns:
        dict: Content of the JSON file
    """
    try:
        data_dir = Path(__file__).parent.parent / 'data' / 'json_files'
        with open(data_dir / filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return {}

def generate_response(query: str, primary_document: str, secondary_documents: List[str], number_of_similar_questions: int = 1) -> str:
    """
    Enhanced response generation using similar questions
    """
    manager = EmbeddingsManager()
    model = None
    response = None
    try:
        # Load document contents
        primary_content = load_json_content(primary_document) if primary_document else {}
        secondary_contents = [load_json_content(doc) for doc in secondary_documents]
        similar_questions = manager.search_similar_questions(query, k=number_of_similar_questions)
        
        similar_contexts = "\n".join([
            f"Similar Question: {q['question']}\nAnswer: {q['answer']}\nMetadata: {q['metadata']}"
            for q in similar_questions
        ])
        
        generation_config = {
            'temperature': 0.3,
            'top_p': 0.9,
            'top_k': 50,
            'max_output_tokens': 2048,
        }
        
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config
        )
        
        prompt = f"""You are a helpful and knowledgeable advisor on Optional Practical Training (OPT) for international students at the University of San Francisco.
        
        Similar questions, their answers, and metadata:
        {similar_contexts}
        
        Primary Information:
        {json.dumps(primary_content, indent=2)}
        
        Additional Context:
        {json.dumps(secondary_contents, indent=2)}
        
        Current Question: "{query}"
        
        Instructions:
        1. Use ALL the provided information to formulate a comprehensive response
        2. If the information contains specific facts, numbers, requirements, or deadlines, preserve them exactly
        3. Focus on answering the user's specific question
        4. Use a conversational but professional tone while maintaining accuracy
        5. If any information is missing or unclear, acknowledge it
        
        Please provide:
        1. A clear, direct answer to the question
        2. Any relevant requirements or deadlines
        3. Important considerations or warnings if applicable
        4. Next steps or recommended actions if relevant
        
        Format the response in a clear, easy-to-read manner.
        """
        
        response = model.generate_content(prompt)
        result = response.text
        return result
    
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        print(error_msg)
        return error_msg
    finally:
        # Explicit cleanup
        if response:
            del response
        if model:
            del model
