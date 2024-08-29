import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch the API key from the environment
api_key = os.getenv("GROQ_API_KEY")

# Set up the Streamlit app
# Add a banner image
banner_image_url = "banner_image.JPG"  # Replace with your actual image URL or local path
st.image(banner_image_url, use_column_width=True)

st.title("Gen AI Q&A with Groq LLM")
st.write("This application uses the Groq LLM to answer questions related to Generative AI.")

# Check if the API key is available
if not api_key:
    st.warning("API key not found. Please ensure it is set in the .env file.")
    st.stop()

# Initialize the LLM
llm = Groq(model="llama3-70b-8192", api_key=api_key)

# Initialize session state to store messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for index, message in enumerate(st.session_state.messages):
    if message.role == "user":
        st.text_area("User", value=message.content, height=50, max_chars=None, key=f"user_{index}", disabled=True)
    elif message.role == "assistant":
        st.text_area("Assistant", value=message.content, height=50, max_chars=None, key=f"assistant_{index}", disabled=True)

# Input for new question
question = st.text_input("Enter your question:")
if st.button("Send"):
    if question:
        # Add user's question to the session state
        user_message = ChatMessage(role="user", content=question)
        st.session_state.messages.append(user_message)

        # Get the response from the LLM
        chat_response = llm.chat(st.session_state.messages)

        # Debugging: Output the entire response to understand its structure
        st.write("Chat Response:", chat_response)

        # Try to extract and format the response
        try:
            # Assuming the response might be a dictionary with a 'choices' key
            assistant_message_content = chat_response.get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception as e:
            st.error(f"Error extracting message content: {e}")
            assistant_message_content = "An error occurred while processing the response."

        # Add LLM's response to the session state
        ai_message = ChatMessage(role="assistant", content=assistant_message_content)
        st.session_state.messages.append(ai_message)

        # Display the new response
        st.text_area("Assistant", value=assistant_message_content, height=50, max_chars=None, key=f"assistant_{len(st.session_state.messages)}", disabled=True)
    else:
        st.warning("Please enter a question.")
