# OPT Chatbot

A chatbot for answering questions about Optional Practical Training (OPT).

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/OPT_chatbot.git
cd OPT_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the template file
cp .env.template .env

# Edit .env file with your actual API keys
nano .env  # or use any text editor
```

4. Add your API keys to the .env file:
```
GEMINI_API_KEY=your_actual_api_key_here
```

## Security Notes

- Never commit your `.env` file to version control
- The `.gitignore` file is set up to exclude `.env` and other sensitive files
- Always use environment variables for API keys and sensitive data
- Keep your API keys secure and never share them

## Usage

```python
from query_matcher import query_opt_knowledge_base

# The API key will be automatically loaded from .env
query = "What are the requirements for STEM OPT Extension?"
matches = query_opt_knowledge_base(query)

for match in matches:
    print(f"Source: {match['file']}")
    print(f"Question: {match['question']}")
    print(f"Relevance: {match['relevance_score']}")
    print(f"Answer: {match['answer']}")
```

## Directory Structure

```
OPT_chatbot/
├── data/                 # JSON knowledge base files
├── query_matcher.py      # Main query matching logic
├── requirements.txt      # Python dependencies
├── .env                 # Your API keys (not in version control)
├── .env.template        # Template for .env file
└── .gitignore          # Git ignore rules
``` 