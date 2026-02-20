# Contextual Compression Retriever
# Improves retrieval quality by compressing documents after retrieval —
# keeping only the relevant content based on the user's query.
# Uses an LLM compressor (LLMChainExtractor) to extract only relevant sentences/paragraphs.

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Sample documents with mixed content (some parts relevant, some not)
docs = [
    Document(page_content="The Grand Canyon is a famous natural site. Photosynthesis is how plants convert light into energy. Many tourists visit every year."),
    Document(page_content="Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen through photosynthesis. The process occurs in chloroplasts."),
    Document(page_content="The Amazon rainforest spans across 9 countries. Photosynthesis is the primary process driving the carbon cycle in forests. Brazil has the largest share."),
    Document(page_content="Climate change affects global weather patterns. Solar energy is captured by plants during photosynthesis. Renewable energy usage is growing."),
    Document(page_content="Cooking involves heat transfer and chemical reactions. Water boils at 100 degrees Celsius at sea level. Many recipes require precise temperatures."),
]

# Step 1: Create FAISS vector store from documents
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 2: Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Step 3: Set up the LLM compressor
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# Step 4: Wrap into Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Step 5: Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for doc in compressed_results:
    print(doc.page_content)  # Only relevant content!
