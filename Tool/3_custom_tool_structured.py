from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv("../.env")

# Concept: Creating Custom Tools via StructuredTool
# Use this when you need more control over the tool's schema, 
# such as detailed field descriptions for the LLM.

# 1. Define the input schema using Pydantic
class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

# 2. Define the tool logic
def multiply_func(a: int, b: int) -> int:
    return a * b

# 3. Create the StructuredTool instance
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="A tool that multiplies two integers.",
    args_schema=MultiplyInput
)

# Demonstrating invocation
print("--- Structured Tool Invocation ---")
result = multiply_tool.invoke({"a": 10, "b": 20})
print(f"Result of 10 * 20: {result}")
print(f"Args Schema: {multiply_tool.args}")
