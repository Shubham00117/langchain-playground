import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env 
load_dotenv("../.env")

# Initialize Groq Chat Model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Invoke the model
response = model.invoke("What is the capital of India?")

# Print the content
print(response.content)
