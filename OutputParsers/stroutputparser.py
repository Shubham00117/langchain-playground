from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define Model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct")
model = ChatHuggingFace(llm=llm)

# Define Chain
prompt = ChatPromptTemplate.from_template("Tell me a fact about {topic}")
chain = prompt | model | StrOutputParser()

# Run directly
print(chain.invoke({"topic": "the sun"}))
