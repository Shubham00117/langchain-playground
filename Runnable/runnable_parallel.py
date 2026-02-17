from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# 1. Define PromptTemplates separately (Best Practice for maintainability)
twitter_prompt = PromptTemplate.from_template("Write a tweet about {topic}")
linkedin_prompt = PromptTemplate.from_template("Write a LinkedIn post about {topic}")

# 2. Define Runnable chains separately using the pipe operator
twitter_chain = twitter_prompt | model | parser
linkedin_chain = linkedin_prompt | model | parser

# 3. Combine into RunnableParallel
parallel_chain = RunnableParallel(
    tweet=twitter_chain,
    linkedin=linkedin_chain
)

# Execution
result = parallel_chain.invoke({'topic': 'AI'})

print("--- Tweet ---")
print(result['tweet'])
print("\n--- LinkedIn ---")
print(result['linkedin'])
