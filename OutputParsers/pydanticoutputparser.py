import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "../.env"))

# Initialize model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.2-1B-Instruct")
model = ChatHuggingFace(llm=llm)

# Define Pydantic model
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="Name of the city the person belongs to")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create prompt template
template = PromptTemplate(
    template="""Extract the person's name, age, and city from the following text: {query}

{format_instructions}

Important: Return ONLY the JSON data with the actual values, not the schema.""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)



# Create chain
chain = template | model | parser

# Check template conditions
print("\nTemplate Variables:")
print(f"  Input: {template.input_variables}")
print(f"  Partial: {list(template.partial_variables.keys())}")
print(f"\nFormatted Template:\n{'-'*60}")
print(template.format(query="John is 25 years old and lives in New York"))
print('-'*60)

# Invoke
try:
    result = chain.invoke({"query": "John is 25 years old and lives in New York"})
    
    # Real-world usage: Work with the Pydantic object directly
    print(f"\nName: {result.name}")
    print(f"Age: {result.age}")
    print(f"City: {result.city}")
    
    # If you need JSON for API response or storage
    # print(result.model_dump_json())
    
except Exception as e:
    print(f"\nError: {e}")