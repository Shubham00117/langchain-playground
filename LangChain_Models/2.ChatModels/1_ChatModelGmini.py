import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("../.env")

# Initialize Chat Model
model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# Invoke the model
response = model.invoke("What is the capital of India?")

# Print the content
print(response.content)