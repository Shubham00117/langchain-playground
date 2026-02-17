from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# 1. Define PromptTemplates separately
prompt1 = PromptTemplate.from_template("Write a joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain the following joke - {text}")

# 2. Define a separate chain for the first part
# Follows the pattern of creating modular runnable blocks
joke_chain = prompt1 | model | parser

# 3. Combine into the final RunnableSequence
# Yes, it is possible! You can pass entire chains into a Sequence.
final_sequence = RunnableSequence(
    joke_chain,  # First block (Output is a string)
    prompt2,     # Second block (Format the string into the prompt)
    model,       # Third block
    parser       # Fourth block
)

# Execution
print(final_sequence.invoke({'topic': 'AI'}))
