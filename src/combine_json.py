import json
from pathlib import Path
from typing import Dict, List

def combine_json_files() -> Dict:
    """
    Combines all JSON files from the json_files directory into a single knowledge base.
    
    Returns:
        Dict: Combined knowledge base with categorized information
    """
    # Get the data directory path
    data_dir = Path(__file__).parent.parent / 'data' / 'json_files'
    
    # Initialize the knowledge base structure
    knowledge_base = {
        "categories": {},
        "metadata": {
            "total_sections": 0,
            "total_qa_pairs": 0,
            "last_updated": "",
            "source_files": []
        }
    }
    
    # Process each JSON file
    for json_file in data_dir.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Get category name from filename
            category_name = json_file.stem.replace('_', ' ').title()
            
            # Add to knowledge base
            knowledge_base["categories"][category_name] = data
            
            # Update metadata
            knowledge_base["metadata"]["source_files"].append(json_file.name)
            if "sections" in data:
                knowledge_base["metadata"]["total_sections"] += len(data["sections"])
            if "qa_pairs" in data:
                knowledge_base["metadata"]["total_qa_pairs"] += len(data.get("qa_pairs", []))
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
    
    # Add timestamp
    from datetime import datetime
    knowledge_base["metadata"]["last_updated"] = datetime.now().isoformat()
    
    # Save the combined knowledge base
    output_path = data_dir.parent / 'knowledge_base.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print(f"Knowledge base created successfully at {output_path}")
    print(f"Total source files: {len(knowledge_base['metadata']['source_files'])}")
    print(f"Total sections: {knowledge_base['metadata']['total_sections']}")
    print(f"Total Q&A pairs: {knowledge_base['metadata']['total_qa_pairs']}")
    
    return knowledge_base

if __name__ == "__main__":
    combine_json_files() 