from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize Local HuggingFace Embeddings
# This will download the model to your machine and run 100% OFFLINE.
# No API key is required.
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 2. Embed a query
text = "Delhi is the capital of India."
vector = embeddings.embed_query(text)

# 3. Print the result
print(f"Embedding Length (all-mpnet-base-v2): {len(vector)}")
print(f"First 5 values: {vector[:5]}")
