import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load .env
load_dotenv("../.env")

# Initialize Hugging Face Model for DeepSeek-R1
# Note: DeepSeek-R1 is a very large model. On the free Hugging Face API, 
# it might experience high latency or temporary unavailability.
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

# Invoke the model
print("Sending request to DeepSeek-R1...")
response = model.invoke("What is the main benefit of using a reasoning model like DeepSeek-R1?")

# Print the content
print(f"\nResponse:\n{response.content}")
