import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Load environment variables (API Keys)
load_dotenv()

# =================================================================
# 1. CREATE DOCUMENTS WITH METADATA (Concept: Document Schema)
# =================================================================
# Each Document consists of 'page_content' and 'metadata' (useful for filtering)
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
# 2. INITIALIZE CHROMA (Concept: Vector Database Persistence)
# =================================================================
# We specify where to save the data (persist_directory) and the embedding function
vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',  # This creates a folder to store the data
    collection_name='ipl_players'
)

# =================================================================
# 3. ADD DOCUMENTS (Concept: Vectorization & Storage)
# =================================================================
# This step converts text to vectors and stores them. It returns custom IDs.
print("\n--- Adding Documents ---")
ids = vector_store.add_documents(docs)
print(f"Added {len(ids)} documents with IDs: {ids[:3]}...")

# =================================================================
# 4. VIEW STORED DATA (Concept: Inspection/Debugging)
# =================================================================
# We can retrieve stored items to verify they were saved correctly
print("\n--- Inspecting Stored Data ---")
stored_data = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print(f"Total entries in DB: {len(stored_data['ids'])}")

# =================================================================
# 5. SIMILARITY SEARCH (Concept: Semantic Retrieval)
# =================================================================
# Searching by 'meaning' rather than just keywords.
print("\n--- Basic Similarity Search ---")
query = "Who among these are a bowler?"
results = vector_store.similarity_search(query, k=2)
for res in results:
    print(f"Result: {res.page_content} [{res.metadata['team']}]")

# =================================================================
# 6. SIMILARITY SEARCH WITH SCORE (Concept: Relevance Scoring)
# =================================================================
# L2 distance: Lower score means more similar.
print("\n--- Search with Relevance Score ---")
results_with_score = vector_store.similarity_search_with_score(query, k=2)
for res, score in results_with_score:
    print(f"Score: {score:.4f} | Content: {res.page_content[:50]}...")

# =================================================================
# 7. METADATA FILTERING (Concept: Hybrid Search)
# =================================================================
# Restricting the search to specific metadata values.
print("\n--- Searching with Metadata Filter (Team: CSK) ---")
filtered_results = vector_store.similarity_search_with_score(
    query="Show me players",
    filter={"team": "Chennai Super Kings"}
)
for res, score in filtered_results:
    print(f"Result: {res.page_content} (Team: {res.metadata['team']})")

# =================================================================
# 8. UPDATE DOCUMENTS (Concept: Data Maintenance)
# =================================================================
# To update, we need the specific ID of the document.
print("\n--- Updating a Document ---")
updated_doc = Document(
    page_content="Virat Kohli, the former captain of RCB, is a legend of the game.",
    metadata={"team": "Royal Challengers Bangalore", "type": "Legend"}
)
vector_store.update_documents(
    ids=[ids[0]], # Using the first ID returned earlier
    documents=[updated_doc]
)
print("Updated the first document.")

# =================================================================
# 9. DELETE DOCUMENTS (Concept: Cleanup)
# =================================================================
# Removing entries using their UUIDs.
print("\n--- Deleting a Document ---")
vector_store.delete(ids=[ids[0]])
print(f"Deleted document with ID: {ids[0]}")

# =================================================================
# 10. VERIFY DELETION
# =================================================================
final_count = len(vector_store.get()['ids'])
print(f"\nFinal count after deletion: {final_count}")
