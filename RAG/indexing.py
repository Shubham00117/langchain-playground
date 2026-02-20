# Stage 1 — Indexing
# Preparing the knowledge base so it can be efficiently searched at query time.
# 4 Sub-steps: Document Ingestion → Text Chunking → Embedding Generation → Vector Store Storage

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Sub-step 1: Document Ingestion — loading source knowledge into memory
# In a real RAG pipeline, you'd use loaders like PyPDFLoader, YoutubeLoader, WebBaseLoader, etc.
# Here we simulate with raw Document objects.
raw_documents = [
    Document(page_content="Photosynthesis is how plants convert light energy into chemical energy. The process occurs in chloroplasts using chlorophyll."),
    Document(page_content="The water cycle involves evaporation, condensation, and precipitation. It is essential for distributing fresh water across the planet."),
    Document(page_content="Mitochondria are the powerhouses of the cell. They produce ATP through cellular respiration."),
    Document(page_content="The Grand Canyon was formed by the Colorado River over millions of years. It is located in Arizona, USA."),
    Document(page_content="DNA stores genetic information using four bases: adenine, thymine, guanine, and cytosine. It forms a double helix structure."),
]

# Sub-step 2: Text Chunking — break large documents into small, semantically meaningful chunks
# Why chunk? LLMs have context limits (4K-32K tokens). Smaller chunks = more focused search results.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(raw_documents)

print(f"Original documents: {len(raw_documents)}")
print(f"After chunking: {len(chunks)} chunks")

# Sub-step 3: Embedding Generation — convert each chunk into a dense vector
embedding_model = OpenAIEmbeddings()

# Sub-step 4: Storage in Vector Store — store vectors + original text + metadata
vectorstore = FAISS.from_documents(chunks, embedding_model)

print(f"\nVector store created with {len(chunks)} vectors")
print("Indexing complete! Ready for retrieval.")
