import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#jsonoutputparser does not support schema enforce
# Load .env
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "../.env"))

# Initialize Model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct")
model = ChatHuggingFace(llm=llm)

# 1. Initialize JsonOutputParser
parser = JsonOutputParser()

# 2. Define Template with format_instruction
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# 3. Create Chain
chain = template | model | parser

# 4. Invoke and Print
result = chain.invoke({'topic': 'black hole'})
print(result)
