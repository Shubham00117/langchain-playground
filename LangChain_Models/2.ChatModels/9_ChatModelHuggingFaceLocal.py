from langchain_huggingface import HuggingFacePipeline

# 1. Initialize the local model using from_model_id
# pipeline_kwargs passes specific generation settings to the underlying pipeline
# When you run this, it will download the 1.5B Qwen model (~3GB)
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
    }
)

# 2. Simple query
response = llm.invoke("What is the capital of India?")

# 3. Print the result
print(response)
