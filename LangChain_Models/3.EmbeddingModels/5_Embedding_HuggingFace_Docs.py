from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load environment variables
load_dotenv("../.env")

# Initialize Hugging Face Online Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# 2. Embed multiple documents
documents = [
    "Delhi is the capital of India.",
    "The sun rises in the east.",
    "LangChain is a great framework for AI."
]
vectors = embeddings.embed_documents(documents)

# Print results
print(f"Number of documents embedded: {len(vectors)}")
print(f"First document embedding length: {len(vectors[0])}")
print(f"First 5 values of first document: {vectors[0][:5]}")
