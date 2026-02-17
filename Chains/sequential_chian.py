import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Model and Parser
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
parser = StrOutputParser()

# 3. Define Prompt Templates
# Prompt 1: Detailed Report
prompt1 = ChatPromptTemplate.from_template("Write a detailed report about {topic}")

# Prompt 2: 5-Pointer Summary (takes 'text' as input)
prompt2 = ChatPromptTemplate.from_template("Give me a 5-pointer summary of the following text:\n\n{text}")

# 4. Create Sequential Chain
# flow: prompt1 → model → parser → lambda (mapping) → prompt2 → model → parser
# Why the lambda? prompt2 expects a dictionary with a 'text' key. 
# The previous step (parser) returns a plain string. 
# Lambda maps that string to the key: {"text": string_output}
chain = prompt1 | model | parser | (lambda x: {"text": x}) | prompt2 | model | parser

# 5. Invoke the Chain
topic = "Generative AI"
print(f"Generating sequential report and summary for: {topic}...\n")
result = chain.invoke({"topic": "Ai Jobs in India"})

# 6. Print Results and Graph
print("--- Final 5-Pointer Summary ---")
print(result)

print("\n--- Chain Graph ---")
chain.get_graph().print_ascii()
