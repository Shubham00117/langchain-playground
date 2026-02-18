from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# Custom function to be used with RunnableLambda
def word_count(text):
    return len(text.split())

# 1. Define Prompt
prompt = PromptTemplate.from_template("Write a short joke about {topic}")

# 2. Joke Generation Chain
joke_gen_chain = RunnableSequence(prompt, model, parser)

# 3. Parallel Chain with RunnableLambda
# RunnableLambda allows you to run custom Python functions within a chain
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})
 
# 4. Final Sequence
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Execution
topic = "Programming"
result = final_chain.invoke({'topic': topic})

print(f"Topic: {topic}")
print(f"Joke: {result['joke']}")
print(f"Word Count: {result['word_count']}")
