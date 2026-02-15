import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load .env
load_dotenv("../.env")

# Initialize Anthropic Claude Model
model = ChatAnthropic(model="claude-3-5-sonnet-latest")

# Invoke the model
response = model.invoke("What is the capital of India?")

# Print the content
print(response.content)
