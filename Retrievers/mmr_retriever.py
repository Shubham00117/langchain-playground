# MMR (Maximal Marginal Relevance) Retriever
# Reduces redundancy in retrieved results while maintaining high relevance.
# Picks results that are relevant to the query AND different from each other.

# lambda_mult=1 → Full relevance (like standard similarity search)
# lambda_mult=0 → Maximum diversity
# lambda_mult=0.5 → Balanced (recommended default)

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Sample documents about climate change (some are similar to each other)
docs = [
    Document(page_content="Climate change is causing glaciers to melt rapidly in the Arctic region."),
    Document(page_content="Glaciers in the Arctic are melting at an alarming rate due to rising temperatures."),
    Document(page_content="Deforestation in the Amazon is accelerating global climate change."),
    Document(page_content="Climate change is increasing the frequency of wildfires in California."),
    Document(page_content="Rising sea levels due to climate change threaten coastal cities."),
]

# Initialize embeddings & create FAISS vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Enable MMR in the retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",              # <-- This enables MMR
    search_kwargs={
        "k": 3,                    # Top results to return
        "lambda_mult": 1          # Relevance-diversity balance
    }
)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
