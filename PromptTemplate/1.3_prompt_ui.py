
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv

# 1. Setup paths relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Load API keys from .env file
# The .env file is in the root directory
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 3. Setup the simple UI text
st.title("Research Paper Explainer 🤖")
st.write("Select options below to generate a paper summary.")

# 4. Create Dropdowns (Selectbox)
paper_input = st.selectbox(
    "Select Paper Name:", 
    ["Attention Is All You Need", "BERT", "GPT-3", "YOLO"]
)

style_input = st.selectbox(
    "Select Explanation Style:", 
    ["Beginner", "Expert", "Funny"]
)

length_input = st.selectbox(
    "Select Length:", 
    ["Short", "Medium"]
)

# 5. Load the Prompt Template from JSON file
json_path = os.path.join(current_dir, "1.2_prompt.json")
prompt = load_prompt(json_path)

# 6. Button to click
if st.button("Summarize"):
    
    # Initialize the Chat Model
    chat = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    # Create the chain using the pipe operator
    chain = prompt | chat
    
    # Get response
    st.write("Generating answer...")
    response = chain.invoke({
        "paper": paper_input, 
        "style": style_input, 
        "length": length_input
    })
    
    # Show the result
    st.write(response.content)
