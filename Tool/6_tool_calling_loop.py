from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# Load environment variables
load_dotenv("../.env")

# Step 1: Define the Tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Step 2: Initialize the LLM and bind the tool
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools([multiply])

# Step 3: Start the conversation
messages = [HumanMessage("What is 3 multiplied by 5?")]

# Step 4: LLM decides which tool to use
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print("--- AI Tool Call ---")
print(ai_msg.tool_calls)  # Shows: [{'name': 'multiply', 'args': {'a': 3, 'b': 5}, ...}]

# Step 5: Execute the tool and feed result back as ToolMessage
for tool_call in ai_msg.tool_calls:
    result = multiply.invoke(tool_call["args"])
    messages.append(
        ToolMessage(content=str(result), tool_call_id=tool_call["id"])
    )

# Step 6: LLM generates final answer using the tool result
final_response = llm_with_tools.invoke(messages)
print("\n--- Final Answer ---")
print(final_response.content)  # → "3 multiplied by 5 is 15."
