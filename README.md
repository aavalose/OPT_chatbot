# OPT Chatbot

A Streamlit-based chatbot powered by Google's Gemini AI for answering questions about Optional Practical Training (OPT) at the University of San Francisco.

## Features

- Interactive chat interface using Streamlit
- AI-powered responses using Google's Gemini model
- Category-based question routing
- Semantic search for similar questions
- Comprehensive OPT information database

## Project Structure

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

# Edit .env file with your API keys
nano .env  # or use any text editor
```

4. Add your API keys to the .env file:
```
```

## Running the Application

From the project root directory, run:
```bash
streamlit run frontend/app.py
```

The application will be available at `http://localhost:8501` by default.

## Security Notes

- Never commit your `.env` file to version control
- The `.gitignore` file is set up to exclude `.env` and other sensitive files
- Always use environment variables for API keys and sensitive data
- Keep your API keys secure and never share them

## Features

The chatbot:
- Categorizes questions into relevant topics
- Searches for similar previous questions
- Provides comprehensive answers using multiple knowledge sources
- Maintains chat history during the session
- Allows clearing chat history

## Dependencies

- streamlit
- google-generativeai
- python-dotenv
- Additional dependencies listed in requirements.txt

## Development

To contribute to this project:
1. Create a new branch for your feature
2. Make your changes
3. Ensure all imports and paths are correctly configured
4. Test the application thoroughly
5. Submit a pull request

## License

This project is currently pending license selection. License details will be added soon.

In the meanwhile, please acknowledge [@aavalose](https://github.com/aavalose) when using or referencing this project.