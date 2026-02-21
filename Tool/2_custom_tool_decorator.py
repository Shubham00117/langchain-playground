from dotenv import load_dotenv
from langchain_core.tools import tool

# Load environment variables
load_dotenv("../.env")

# Concept: Creating Custom Tools via @tool Decorator
# This is the simplest way to create a tool. 
# LangChain uses the function name, docstring, and type hints to create the tool schema.

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Demonstrating tool attributes
print("--- Tool Metadata ---")
print(f"Name: {multiply.name}")
print(f"Description: {multiply.description}")
print(f"Args Schema: {multiply.args}")

# Invoking the tool
print("\n--- Tool Invocation ---")
result = multiply.invoke({"a": 3, "b": 5})
print(f"Result of 3 * 5: {result}")
