from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load .env
load_dotenv("../.env")

# Initialize LLM (Traditional LLM style - takes string, returns string)
llm = GoogleGenerativeAI(model="gemini-flash-latest")

# Simple Chain
chain=llm

# Invoke and print
response = chain.invoke("what is capital of India ?")
print(response)