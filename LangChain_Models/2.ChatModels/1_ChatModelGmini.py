from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("../.env")

# Initialize Chat Model
model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# Invoke the model
response = model.invoke("What is the square root of 144 ?")

# Print the content
print(response.content)