from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# 1. Define PromptTemplates
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Using 'text' as the variable name to match screenshot logic
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# 2. Report Generation Chain
report_gen_chain = RunnableSequence(prompt1, model, parser)

# 3. Branching Logic
# Condition: If word count > 500, run summarization. Else, pass the text as is.
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

# 4. Final Sequence
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Execution
topic = "The history of Space Exploration"
print(f"--- Generating Report for: {topic} ---")
result = final_chain.invoke({'topic': topic})

print("\n--- Final Output ---")
print(result)
