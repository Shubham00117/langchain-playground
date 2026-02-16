import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "../.env"))

# Model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct")
model = ChatHuggingFace(llm=llm)

# Parser
parser = StrOutputParser()

# Prompt 1 → Detailed report
prompt1 = ChatPromptTemplate.from_template(
    "Write a detailed report about {topic}"
)

# Prompt 2 → Summary (takes output of prompt1)
prompt2 = ChatPromptTemplate.from_template(
    "Summarize this text in 2 sentences:\n{input}"
)

# 🔗 Single Chain with Two Prompts
chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser

# Run
result = chain1.invoke({"topic": "Quantum Computing"})
summary = chain2.invoke({"input": result})

print("\n--- Detailed Report ---")
print(result)
print("\n--- Final Summary ---")
print(summary)