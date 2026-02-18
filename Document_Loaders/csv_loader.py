import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
# Robust way to find the .env file in the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# 2. Robust File Path Handling (Industry Standard)
# Pathlib treats paths as objects, making them more robust across Windows/Mac/Linux
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "sample_products.csv"

# Load CSV Data
loader = CSVLoader(file_path=str(file_path))
data = loader.load()

# Combine all rows into a single text block for the LLM
context = "\n".join([doc.page_content for doc in data])

# 3. Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# 4. Define Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the provided CSV data:
{csv_data}

Question: {question}
""")

# 5. Chain and Execute
chain = prompt | llm | StrOutputParser()

print("--- LLM ANSWER ---")
print(chain.invoke({
    "csv_data": context,
    "question": "What is the price of the iPhone 17 Pro Max?"
}))
