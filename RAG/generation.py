# Stage 4 — Generation
# The final step where the LLM uses the augmented prompt (query + retrieved context)
# to generate a grounded, accurate response.

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Build the index (Stage 1)
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# The RAG prompt template
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question ONLY from the provided context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""")

# Complete RAG Chain — all 4 stages in one LCEL chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Query the full RAG pipeline
query = "What is photosynthesis?"
response = rag_chain.invoke(query)

print(f"Query: {query}")
print(f"\nResponse:\n{response}")
