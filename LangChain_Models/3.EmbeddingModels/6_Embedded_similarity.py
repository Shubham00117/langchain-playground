import math
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load variables
load_dotenv("../.env")
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# 5 lines of documents
documents = [
    "Python is a versatile programming language for AI.",
    "Delhi is the capital and a major city of India.",
    "The sun provides energy for all life on Earth.",
    "Cricket is a popular sport played with a bat and ball.",
    "Deep learning is a subset of machine learning."
]

query = "Tell me about the capital city of India."

# Generate embeddings
doc_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)


# Simple Similarity Logic
def calculate_score(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (mag1 * mag2)

# Get all scores
scores = [calculate_score(query_vector, v) for v in doc_vectors]

# Use the requested sorting logic to find the best match
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Print only the final result
print(documents[index])
