import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 1. Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Models and Parser
# Model 1: Groq
model1 = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# Model 2: Hugging Face
hf_llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct")
model2 = ChatHuggingFace(llm=hf_llm)

parser = StrOutputParser()

# 3. Define Prompt Templates
# prompt1 and prompt2 will run in parallel on the same input {text}
prompt1 = PromptTemplate.from_template("Write concise study notes about {text}")
prompt2 = PromptTemplate.from_template("Generate 3 MCQs about {text}")

# prompt3 will merge the results of prompt1 and prompt2
prompt3 = PromptTemplate.from_template(
    "Merge the provided notes and quiz into a single document\n\n"
    "notes -> {notes}\n\n"
    "quiz -> {quiz}"
)

# 4. Create Parallel Chain (Logic from image)
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

# 5. Create Merge Chain
merge_chain = prompt3 | model1 | parser

# 6. Combine into Final Chain
chain = parallel_chain | merge_chain

# 7. Run the Chain
text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior. 
Modern AI is dominated by Machine Learning (ML), where algorithms learn patterns from data. 
One of the most exciting recent developments is Generative AI, which can create new content like text, images, and code. 
These systems are trained on massive datasets and use neural networks to understand and generate human-like responses.
"""
print(f"Executing parallel chain for the provided text...\n")
result = chain.invoke({"text": text})

# 8. Output Result and Graph
print("--- Final Merged Output ---")
print(result)

print("\n--- Chain Graph ---")
chain.get_graph().print_ascii()
