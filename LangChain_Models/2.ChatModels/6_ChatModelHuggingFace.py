import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load .env 
load_dotenv("../.env")

# Initialize Hugging Face Model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Invoke the model
response = model.invoke("What is the capital of India?")

# Print the content
print(response.content)