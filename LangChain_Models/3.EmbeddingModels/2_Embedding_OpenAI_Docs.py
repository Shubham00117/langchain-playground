from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load .env
load_dotenv("../.env")

# 1. Initialize OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Embed multiple documents
documents = [
    "Delhi is the capital of India.",
    "The sun rises in the east.",
    "LangChain is a great framework for AI."
]
vectors = embeddings.embed_documents(documents)

# 3. Print the result
print(f"Number of documents embedded: {len(vectors)}")
print(f"First document embedding length: {len(vectors[0])}")
