import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv("../.env")

# Initialize OpenAI Model
model = ChatOpenAI(model="gpt-4o")

# Invoke the model
response = model.invoke("What is the capital of India?")

# Print the content
print(response.content)
