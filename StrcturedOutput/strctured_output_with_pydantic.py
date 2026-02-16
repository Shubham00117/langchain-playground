import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# Fetch API key using specific path logic
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# Define output schema using Pydantic
class ReviewAnalysis(BaseModel):
    summary: str = Field(description="A brief summary of the entire review")
    pros: List[str] = Field(description="A list of positive features mentioned")
    cons: List[str] = Field(description="A list of negative features or drawbacks")
    positive: List[str] = Field(description="Specific positive words or phrases used in the review")
    negative: List[str] = Field(description="Specific negative words or phrases used in the review")
    sentiment: str = Field(description="Overall emotional tone of the review (e.g., positive, negative, neutral, mixed)")
    key_theme: str = Field(description="The primary subject or topic of the review")

# Sample iPhone 17 Pro Max review (Expanded)
review_text = """
The iPhone 17 Pro Max is finally here, and it’s a bold leap forward for Apple. 
The standout feature is undoubtedly the new under-display camera technology, 
which eliminates the Dynamic Island and provides a truly immersive, edge-to-edge display. 
The ProMotion 3.0 screen is brighter than ever, reaching a peak of 3500 nits, making it perfect for outdoor use.

Under the hood, the A19 chip is a performance beast, handling intensive 4K video editing and 
AAA gaming with zero lag. However, that power comes with a physical cost; the phone is 
noticeably heavier and bulkier than the 16 Pro Max, making it a bit tiring to hold during long sessions.

The camera system has seen a massive upgrade with a 72MP main sensor that captures incredible detail, 
though the software processing can sometimes over-sharpen images in low light. 
Battery life is legendary, easily lasting two full days for moderate users. 
A major disappointment, however, remains the charging speed—stuck at 35W while competitors 
are hitting 100W+. Plus, the $1,299 starting price is a tough pill to swallow for many.
Overall, it's a masterpiece of engineering with a few lingering frustrations.
"""

# Initialize Groq model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Bind schema to model
structured_llm = llm.with_structured_output(ReviewAnalysis)

# Extract structured data
if __name__ == "__main__":
    result = structured_llm.invoke(review_text)
    
    # Print as simple key-value pairs
    print("--- Review Analysis Results ---")
    # Convert Pydantic model to dict to keep the loop logic
    for key, value in result.model_dump().items():
        print(f"{key.upper()}: {value}")
