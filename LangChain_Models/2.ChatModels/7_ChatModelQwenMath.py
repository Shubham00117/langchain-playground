import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load .env
load_dotenv("../.env")

# Initialize Hugging Face Math Model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Math-1.5B",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Invoke the model with a math question
response = model.invoke("What is the square root of 144 ?")

# Print the content
print(response.content)
