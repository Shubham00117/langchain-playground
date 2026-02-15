from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load .env
load_dotenv("../.env")

# 1. Initialize OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Embed a single query
text = "Delhi is the capital of India."
vector = embeddings.embed_query(text)

# 3. Print the result
print(f"Embedding Length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
