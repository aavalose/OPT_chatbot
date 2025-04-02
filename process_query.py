from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
from pathlib import Path
from typing import List, Tuple

# Load environment variables first
load_dotenv()
print(f"API Key loaded: {os.getenv('GEMINI_API_KEY') is not None}")  # Debug line

# Configure the Gemini API with your key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def preprocess_query(query: str) -> List[str]:
    """
    Analyze the query and return relevant categories.
    
    Args:
        query (str): The user's question about OPT
        
    Returns:
        List[str]: List of relevant categories
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Analyze this question and return up to THREE most relevant categories from the following list, ordered by relevance:
        [Your existing prompt content...]
        """
        
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
        return validated_categories if validated_categories else ['Other']
        
    except Exception as e:
        print(f"Error in preprocessing query: {str(e)}")
        return ['Other']

def match_files(categories: List[str]) -> Tuple[str, List[str]]:
    """
    Match categories to corresponding JSON files.
    
    Args:
        categories (List[str]): List of categories to match
        
    Returns:
        Tuple[str, List[str]]: Primary document and list of secondary documents
    """
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

def generate_response(query: str, primary_document: str, secondary_documents: List[str]) -> str:
    """
    Generate a response using the query and relevant documents.
    
    Args:
        query (str): The user's question
        primary_document (str): Primary JSON file to use
        secondary_documents (List[str]): Additional JSON files to reference
        
    Returns:
        str: Generated response
    """
    try:
        # Load document contents
        primary_content = load_json_content(primary_document) if primary_document else {}
        secondary_contents = [load_json_content(doc) for doc in secondary_documents]
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""You are an expert advisor on Optional Practical Training (OPT) for international students.
        Use the following information to provide a comprehensive and accurate answer.
        
        Primary Information:
        {primary_content}
        
        Additional Context:
        {secondary_contents}
        
        Question: {query}
        
        Please provide:
        1. A clear, direct answer to the question
        2. Any relevant requirements or deadlines
        3. Important considerations or warnings if applicable
        4. Next steps or recommended actions if relevant
        
        Format the response in a clear, easy-to-read manner.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}"
        print(error_msg)  # For debugging
        return error_msg 