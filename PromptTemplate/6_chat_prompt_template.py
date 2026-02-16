import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Model
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)

# 3. Define ChatPromptTemplate
# You can use tuples (role, template_string)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}"),
])

# 4. Format the prompt with variables
prompt_value = template.invoke({
    "input_language": "English",
    "output_language": "French",
    "text": "I love programming with LangChain!"
})

# 5. Invoke LLM
response = model.invoke(prompt_value)

# 6. Print result
print(response.content)
