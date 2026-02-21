from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Load environment variables
load_dotenv("../.env")

# Concept: Creating Custom Tools via BaseTool Class
# This is the most flexible approach, allowing for complex logic and state.

# 1. Define the input schema
class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

# 2. Inherit from BaseTool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "A tool that multiplies two integers using a class-based approach."
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        """The heart of the tool - where the logic lives."""
        return a * b

# 3. Instantiate and use
multiply_tool = MultiplyTool()

print("--- BaseTool Class Invocation ---")
result = multiply_tool.invoke({"a": 7, "b": 6})
print(f"Result of 7 * 6: {result}")
print(f"Name: {multiply_tool.name}")
