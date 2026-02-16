import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# 1. Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Model
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# 3. Create Initial List with System and Human Messages
messages = [
    SystemMessage(content="You are a helpful and funny assistant."),
    HumanMessage(content="Tell me a defination of langchain ?")
]

# 4. Invoke LLM
response = model.invoke(messages)

# 5. Append AI Response to List
messages.append(response)

# 6. Print Final List
print(messages)
