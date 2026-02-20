# Wikipedia Retriever
# A retriever that queries the Wikipedia API to fetch relevant content for a given query.
# No local document setup needed — the data source is Wikipedia itself.

# Install: pip install wikipedia langchain-community
from langchain_community.retrievers import WikipediaRetriever

# Initialize retriever with top_k and language
retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en"
)

# Define your query
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")  # truncate for display
