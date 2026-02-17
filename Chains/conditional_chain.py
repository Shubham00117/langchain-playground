import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

# 1. Load Environment Variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 2. Setup Model
model = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# 3. Define Pydantic Class for Sentiment Classification
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment of the feedback, either positive, negative, or neutral"
    )

# 4. Setup Pydantic Parser
parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

# 5. Create Classification Chain
classification_prompt = PromptTemplate(
    template="Analyze the sentiment of feedback.\n{format_instructions}\nFeedback: {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
classification_chain = classification_prompt | model | parser

# 6. Define Branch Response Chains
positive_response_chain = PromptTemplate.from_template(
    "The user gave positive feedback: {feedback}. Thank them."
) | model | StrOutputParser()

negative_response_chain = PromptTemplate.from_template(
    "The user gave negative feedback: {feedback}. Apologize and offer help."
) | model | StrOutputParser()

# 7. Create the Conditional Branch
branch = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive", positive_response_chain),
    (lambda x: x["sentiment"].sentiment == "negative", negative_response_chain),
    lambda x: "couldnot find the sentiment"
)

# 8. Assemble Full Chain
# Input: {"feedback": "text"} 
# Map to: {"sentiment": classification_result, "feedback": original_text}
# Then branch.
full_chain = (
    {"sentiment": classification_chain, "feedback": lambda x: x["feedback"]} 
    | branch
)

# 9. Run the Chain
feedback = "I absolutely love this product! It's life-changing."

print(f"\n--- Testing Feedback: {feedback} ---")
result = full_chain.invoke({"feedback": feedback})
print(f"Response: {result}")

print("\n--- Chain Graph ---")
full_chain.get_graph().print_ascii()
