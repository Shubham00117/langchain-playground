
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load Environment Variables
# The .env file is in the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Chat Model
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# 3. Simple list to store history
chat_history = [] 

print("System: Chatbot is ready! Type 'exit' to stop.")

while True:
    try:
        # Get input
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
            
        # Add to history
        chat_history.append(HumanMessage(content=user_input))
        
        # Get response (pass the whole history)
        response = model.invoke(chat_history)
        
        # Show and save response
        print(f"AI: {response.content}")
        chat_history.append(AIMessage(content=response.content))
        
    except Exception as e:
        print(f"Error: {e}")
        break  
