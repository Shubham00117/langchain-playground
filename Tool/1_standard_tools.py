from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

# Load environment variables (API keys, etc.)
load_dotenv("../.env")

# Concept: Built-in Standard Tools
# These are production-ready tools provided by LangChain.

# 1. DuckDuckGo Search Tool
print("--- DuckDuckGo Search ---")
search_tool = DuckDuckGoSearchRun()
try:
    results = search_tool.invoke("top news in india today")
    print(results)
except Exception as e:
    print(f"Error in Search Tool: {e}")

# 2. Shell Tool
print("\n--- Shell Tool ---")
shell_tool = ShellTool()
try:
    # Note: Shell tool can be dangerous, use with caution!
    results = shell_tool.invoke("whoami")
    print(f"Result of 'whoami': {results}")
except Exception as e:
    print(f"Error in Shell Tool: {e}")
