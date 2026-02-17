from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
model = ChatGroq(model="llama-3.3-70b-versatile")
prompt = PromptTemplate.from_template("Tell me a short fun fact about {topic}")
parser = StrOutputParser()

# --- LCEL (LangChain Expression Language) ---
# The core idea of LCEL is using the pipe operator (|) to chain runnables.
# Data flows from: Input -> Prompt -> Model -> Parser -> Output
# The pipe operator (|) creates a RunnableSequence under the hood.
chain = prompt | model | parser

# Execution
print("--- LCEL Chain Demo ---")
result = chain.invoke({"topic": "Penguins"})
print(result)
