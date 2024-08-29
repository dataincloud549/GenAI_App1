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
banner_image_url = "banner_image.JPG"
 # Replace with your actual image URL or local path
st.image(banner_image_url, use_column_width=True)

st.title("Gen AI Q&A with Groq LLM")
st.write("This application uses the Groq LLM to answer questions related to Generative AI.")

# Check if the API key is available
if not api_key:
    st.warning("API key not found. Please ensure it is set in the .env file.")
    st.stop()

# Initialize the LLM
llm = Groq(model="llama3-70b-8192", api_key=api_key)

# System message to guide the LLM's behavior
system_message = ChatMessage(
    role="system",
    content=(
        "You are an AI instructor specialized in Generative AI. "
        "You should only answer questions related to Generative AI, "
        "such as GPT models, transformers, neural networks, AI ethics, "
        "and applications of AI in creativity and research. If a user "
        "asks a question that is not related to Generative AI, kindly inform "
        "them that you can only answer Generative AI-related questions."
    )
)

# Basic completion example
st.subheader("Ask a Question about Generative AI")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        # Build the messages list including the system message
        messages = [system_message, ChatMessage(role="user", content=question)]
        
        # Get the response from the LLM
        response = llm.chat(messages)
        
        # Display the response
        st.write(response)
    else:
        st.warning("Please enter a question.")
