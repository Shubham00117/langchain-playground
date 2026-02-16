from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env
load_dotenv("../.env")

# Initialize Groq Model with parameters
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=1.1,
    max_completion_tokens=50
)

# Invoke the model
response = model.invoke("Write a 2-line poem about a cat in space.")

print(response.content)
