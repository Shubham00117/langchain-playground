import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load Data
url = "https://www.financialexpress.com/life/technology-iphone-17-iphone-17-pro-iphone-17-pro-max-prices-in-india-vs-us-singapore-canada-and-australia-in-2026-4130592/"

# Set USER_AGENT for compliance and to avoid blocks
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Load Data following industry standard (Explicit initialization)
loader = WebBaseLoader(web_paths=(url,))
docs = loader.load()

# Initialize Components
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

# Prompt logic from the screenshot
prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"]
)

# Chain and Execution
chain = prompt | model | parser

print(chain.invoke({
    "question": "what is the name of product?",
    "text": docs[0].page_content[:6000]
}))
