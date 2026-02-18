import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "../.env"))
file_path = os.path.join(current_dir, "sample_cricket_10pages.pdf")

# 2. Load
loader = PyPDFLoader(file_path)
pages = loader.load()
all_text = "\n".join([p.page_content for p in pages])

# 3. LLM Setup
model = ChatGroq(model="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_template("List the 10 topics from this text:\n\n{text}")
chain = prompt | model | StrOutputParser()

# 4. Simple Output
print(f"Total Pages: {len(pages)}")

print("\n--- METADATA ---")
print(pages[0].metadata)

print("\n--- TOPICS (LLM) ---")
print(chain.invoke({"text": all_text}))

print("\n--- ALL CONTENT ---")
print(all_text)
