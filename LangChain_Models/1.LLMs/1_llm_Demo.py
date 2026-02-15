import warnings
import os

# Suppress the Pydantic V1 warning (compatibility issue with Python 3.14)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.v1.fields")
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Load .env
load_dotenv("../.env")

# Initialize LLM (Traditional LLM style - takes string, returns string)
llm = GoogleGenerativeAI(model="gemini-flash-latest")

# Simple Chain
chain = llm | StrOutputParser()

# Invoke and print
response = chain.invoke("What is the capital of India?")
print(response)