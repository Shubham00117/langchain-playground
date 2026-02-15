from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Load environment variables
load_dotenv("../.env")

# Initialize Hugging Face Online Embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# Embed a single query
text = "Delhi is the capital of India."
vector = embeddings.embed_query(text)


# Print results
print(f"Vector (length {vector}):")

