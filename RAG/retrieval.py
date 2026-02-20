# Stage 2 — Retrieval
# Finding the most relevant pieces of information from the pre-built index.
# Steps: Query → Embed Query → Semantic Search in Vector Store → Rank → Return Top-K Chunks

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# First, build the index (Stage 1)
raw_documents = [
    Document(page_content="Photosynthesis is how plants convert light energy into chemical energy. The process occurs in chloroplasts using chlorophyll."),
    Document(page_content="The water cycle involves evaporation, condensation, and precipitation. It is essential for distributing fresh water across the planet."),
    Document(page_content="Mitochondria are the powerhouses of the cell. They produce ATP through cellular respiration."),
    Document(page_content="The Grand Canyon was formed by the Colorado River over millions of years. It is located in Arizona, USA."),
    Document(page_content="DNA stores genetic information using four bases: adenine, thymine, guanine, and cytosine. It forms a double helix structure."),
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(raw_documents)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Stage 2: Retrieval — convert vectorstore into a retriever
# The retriever uses the SAME embedding model to embed the query
# then performs semantic search (cosine similarity) to find the most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Query the retriever
query = "How do plants produce energy?"
results = retriever.invoke(query)

print(f"Query: '{query}'")
print(f"Retrieved {len(results)} relevant chunks:\n")

for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()
