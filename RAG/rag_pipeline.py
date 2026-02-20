# Complete RAG Pipeline — End to End
# All 4 stages combined: Indexing → Retrieval → Augmentation → Generation
# This file demonstrates the full RAG architecture from first principles.

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# STAGE 1: INDEXING
# Load → Chunk → Embed → Store in Vector DB
# ============================================================

# Sub-step 1: Document Ingestion (simulated with raw documents)
raw_documents = [
    Document(page_content="Photosynthesis is how plants convert light energy into chemical energy. The process occurs in chloroplasts using chlorophyll."),
    Document(page_content="The water cycle involves evaporation, condensation, and precipitation. It is essential for distributing fresh water across the planet."),
    Document(page_content="Mitochondria are the powerhouses of the cell. They produce ATP through cellular respiration."),
    Document(page_content="The Grand Canyon was formed by the Colorado River over millions of years. It is located in Arizona, USA."),
    Document(page_content="DNA stores genetic information using four bases: adenine, thymine, guanine, and cytosine. It forms a double helix structure."),
]

# Sub-step 2: Text Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(raw_documents)

# Sub-step 3 & 4: Embedding Generation + Vector Store Storage
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding_model)

print(f"[Indexing] {len(raw_documents)} docs → {len(chunks)} chunks → stored in FAISS")

# ============================================================
# STAGE 2: RETRIEVAL
# Embed query → Semantic search → Rank → Return Top-K chunks
# ============================================================

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ============================================================
# STAGE 3: AUGMENTATION
# Combine retrieved context + query → enriched prompt
# ============================================================

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question ONLY from the provided context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""")

# ============================================================
# STAGE 4: GENERATION
# LLM generates response grounded in context
# ============================================================

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Run the full pipeline
query = "What is photosynthesis?"
response = rag_chain.invoke(query)

print(f"\n[Query] {query}")
print(f"\n[Response]\n{response}")
