import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Load Environment Variables
# The .env file is in the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Groq AI Model
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
parser = StrOutputParser()

# 3. Define a Simple Prompt Template
prompt = ChatPromptTemplate.from_template("Tell me a fun fact about {topic}")

# 4. Create a Simple Chain
# prompt → model → extract string from response
chain = prompt | model | parser

# 5. Run the Chain
result = chain.invoke({"topic": "Langchian"})

# 6. Print the Result
print(result)
chain.get_graph().print_ascii()
