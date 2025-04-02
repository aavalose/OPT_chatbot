import streamlit as st
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.process_query import preprocess_query, match_files, generate_response

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.title("OPT Assistant Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about OPT..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process the query
                categories = preprocess_query(prompt)
                primary_doc, secondary_docs = match_files(categories)
                response = generate_response(
                    query=prompt,
                    primary_document=primary_doc,
                    secondary_documents=secondary_docs,
                    number_of_similar_questions=2
                )
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add a clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []

if __name__ == "__main__":
    main()
