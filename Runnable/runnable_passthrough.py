from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# 1. Define PromptTemplates
prompt1 = PromptTemplate.from_template("Write a joke about {topic}")
prompt2 = PromptTemplate.from_template("Explain the following joke - {joke}")

# 2. Define the Joke Generation Chain
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# 3. Define the Parallel Chain with Passthrough
# RunnablePassthrough() takes the output of joke_gen_chain (the joke string) 
# and passes it directly to the 'joke' key.
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

# 4. Final Sequence: joke -> (original joke & explanation)
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Execution
result = final_chain.invoke({'topic': 'AI'})

# Printing joke and explanation separately
print("--- THE JOKE ---")
print(result['joke'])

print("\n--- THE EXPLANATION ---")
print(result['explanation'])
