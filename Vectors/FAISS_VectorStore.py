import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# =================================================================
# 1. CREATE DOCUMENTS WITH METADATA
# =================================================================
doc1 = Document(
    page_content="Virat Kohli is one of the most successful batsmen in IPL history, playing for RCB.",
    metadata={"team": "Royal Challengers Bangalore", "type": "Batsman"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians.",
    metadata={"team": "Mumbai Indians", "type": "Captain"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has guided CSK to multiple titles.",
    metadata={"team": "Chennai Super Kings", "type": "Captain"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is a world-class death bowler playing for Mumbai Indians.",
    metadata={"team": "Mumbai Indians", "type": "Bowler"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a top-tier all-rounder for Chennai Super Kings.",
    metadata={"team": "Chennai Super Kings", "type": "All-rounder"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# =================================================================
# 2. INITIALIZE FAISS (Concept: Local Vector Index) 
# =================================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS is often initialized from documents directly
print("\n--- Initializing FAISS and Adding Documents ---")
vector_store = FAISS.from_documents(docs, embeddings)
print("FAISS index created with initial documents.")

# =================================================================
# 3. ADD MORE DOCUMENTS (Concept: Index Expansion)
# =================================================================
doc6 = Document(
    page_content="Suryakumar Yadav is known for his 360-degree batting for Mumbai Indians.",
    metadata={"team": "Mumbai Indians", "type": "Batsman"}
)
print("\n--- Adding New Document ---")
new_ids = vector_store.add_documents([doc6])
print(f"Added new document with ID: {new_ids}")

# =================================================================
# 4. VIEW STORED DATA (Note: FAISS inspection is limited)
# =================================================================
# FAISS doesn't have a .get() like Chroma, but we can see the docstore size
print("\n--- Inspecting Stored Data ---")
print(f"Total entries in FAISS docstore: {len(vector_store.docstore._dict)}")

# =================================================================
# 5. SIMILARITY SEARCH (Concept: Semantic Retrieval)
# =================================================================
print("\n--- Basic Similarity Search ---")
query = "Who among these are a bowler?"
results = vector_store.similarity_search(query, k=2)
for res in results:
    print(f"Result: {res.page_content} (Team: {res.metadata.get('team')})")

# =================================================================
# 6. SIMILARITY SEARCH WITH SCORE (Concept: Relevance Scoring)
# =================================================================
# FAISS uses L2 distance (lower is better)
print("\n--- Search with Relevance Score ---")
results_with_score = vector_store.similarity_search_with_score(query, k=2)
for res, score in results_with_score:
    print(f"Score: {score:.4f} | Content: {res.page_content[:50]}...")

# =================================================================
# 7. METADATA FILTERING
# =================================================================
# FAISS support for filtering varies by version/type. 
# In langchain-community, FAISS supports filtering via a dictionary or callable.
print("\n--- Searching with Metadata Filter (Team: CSK) ---")
filtered_results = vector_store.similarity_search(
    query="Show me players",
    k=2,
    filter={"team": "Chennai Super Kings"}
)
for res in filtered_results:
    print(f"Result: {res.page_content} (Team: {res.metadata.get('team')})")

# =================================================================
# 8. PERSISTENCE (Concept: Local Saving)
# =================================================================
print("\n--- Persisting FAISS Index ---")
vector_store.save_local("my_faiss_index")
print("Index saved to 'my_faiss_index' folder.")

# To load:
#new_db = FAISS.load_local("my_faiss_index", embeddings, allow_dangerous_deserialization=True)

# =================================================================
# 9. DELETE DOCUMENTS (Concept: Cleanup)
# =================================================================
# FAISS deletion requires IDs. Let's delete the one we just added.
print("\n--- Deleting a Document ---")
vector_store.delete([new_ids[0]])
print(f"Deleted document with ID: {new_ids[0]}")
print(f"Total entries now: {len(vector_store.docstore._dict)}")
