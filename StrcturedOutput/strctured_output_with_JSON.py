import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Fetch API key using specific path logic
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# Define output schema using JSON schema (dict)
json_schema = {
    "title": "ReviewAnalysis",
    "description": "Extract structured information from a product review",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string", 
            "description": "A brief summary of the entire review"
        },
        "pros": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "A list of positive features mentioned"
        },
        "cons": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "A list of negative features or drawbacks"
        },
        "positive": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Specific positive words or phrases used in the review"
        },
        "negative": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Specific negative words or phrases used in the review"
        },
        "sentiment": {
            "type": "string", 
            "description": "Overall emotional tone of the review (e.g., positive, negative, neutral, mixed)"
        },
        "key_theme": {
            "type": "string", 
            "description": "The primary subject or topic of the review"
        }
    },
    "required": ["summary", "pros", "cons", "positive", "negative", "sentiment", "key_theme"]
}

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

# Bind schema to model using JSON Schema
structured_llm = llm.with_structured_output(json_schema)

# Extract structured data
if __name__ == "__main__":
    result = structured_llm.invoke(review_text)
    
    # Print as simple key-value pairs
    print("--- Review Analysis Results (JSON Schema) ---")
    # Result is a standard dictionary
    for key, value in result.items():
        print(f"{key.upper()}: {value}")
