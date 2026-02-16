import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Model
model = ChatGroq(model_name="llama-3.3-70b-versatile")

# 3. Create Prompt Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# 4. Load Chat History (Simple logic from image)
chat_history = []
with open('7.2_history.txt') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

# 5. Invoke (Create Prompt)
chain = chat_template | model

response = chain.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund?'
})

print(response.content)
