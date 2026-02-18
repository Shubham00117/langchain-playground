import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
# Getting the absolute path to the project root for the .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Load the Document
# Path to the cricket.txt file
file_path = os.path.join(current_dir, "cricket.txt")
loader = TextLoader(file_path)
docs = loader.load()

# Extracting the raw text from the document "as it is"
# Since TextLoader loads the entire file into one document, we take the first one.
full_text = docs[0].page_content

# 3. Setup LLM and Prompt
model = ChatGroq(model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    "You are a sports journalist. Write a short, engaging 3-sentence summary of the following text:\n\n{text}"
)

# 4. Create Chain and Invoke
chain = prompt | model | StrOutputParser()

print("Loading document and generating summary...\n")
summary = chain.invoke({"text": full_text})

# 5. Output Result

print("--- FULL TEXT ---") 
print(full_text)

print("\n--- METADATA ---")
print(docs[0].metadata)

print("\n--- SUMMARY ---")
print(summary)
