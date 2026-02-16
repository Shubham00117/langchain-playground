from langchain_huggingface import HuggingFacePipeline

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
