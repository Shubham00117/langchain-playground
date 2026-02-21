from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import List

# Load environment variables
load_dotenv("../.env")

# Concept: Toolkits
# A toolkit is a collection of related tools bundled together.

# 1. Mocking tools for the toolkit
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# 2. Creating a Custom Toolkit
class MathToolkit:
    """A toolkit containing basic math tools."""
    def get_tools(self) -> List:
        return [add, multiply]

# 3. Usage
print("--- Using a Custom Toolkit ---")
toolkit = MathToolkit()
tools = toolkit.get_tools()

print(f"Number of tools in toolkit: {len(tools)}")
for t in tools:
    print(f" - {t.name}: {t.description}")

# Concept: Pre-built Toolkits (Example with GoogleDriveToolkit structure)
# Note: This requires specific setup/auth not included here.
# from langchain_community.agent_toolkits import GoogleDriveToolkit
# toolkit = GoogleDriveToolkit(api_client=client)
# tools = toolkit.get_tools()
