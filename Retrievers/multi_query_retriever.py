# Multi-Query Retriever
# Uses an LLM to generate multiple semantically different versions of your query.
# Each sub-query is sent to the retriever independently, then results are combined and deduplicated.
# Solves the problem of a single query missing documents that use different phrasing.

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Sample documents
docs = [
    Document(page_content="Eating a balanced diet with fruits and vegetables is essential for maintaining good health."),
    Document(page_content="Regular physical exercise helps improve cardiovascular fitness and overall well-being."),
    Document(page_content="Stress management techniques like meditation can significantly boost mental health."),
    Document(page_content="Getting 7-8 hours of quality sleep each night is crucial for body recovery."),
    Document(page_content="Drinking sufficient water throughout the day keeps the body hydrated and energized."),
]

# Create FAISS vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

# Create base similarity retriever
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Wrap it with MultiQueryRetriever using an LLM
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatOpenAI(model="gpt-3.5-turbo")
)

# Query
query = "How to improve energy levels and maintain balance?"
results = multiquery_retriever.invoke(query)

print(f"Retrieved {len(results)} unique documents")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
