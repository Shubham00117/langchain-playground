# LangChain Models Notes

## 1. LLMs
* **Core Concept**: Traditional string-in, string-out interface, distinct from Chat Models.
* **Code Topic**: `GoogleGenerativeAI`
```python
# Traditional LLM style - takes string, returns string
llm = GoogleGenerativeAI(model="gemini-flash-latest")
response = llm.invoke("write selenium page class")
```


## 2. Chat Models
* **Unified Interface**: All models use the standard `.invoke()` method, making it easy to swap providers.
* **Code Topic**: **Multi-Provider Setup**
```python
# Google
model = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# Groq (Llama 3) with Custom Params
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=1.1)

# Hugging Face (Serverless Endpoint)
llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-1.5B-Instruct")
model = ChatHuggingFace(llm=llm)

# Transformers Pipeline (Local Execution)
# Downloads model (~3GB) and runs offline
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512}
)
```

## 3. Embedding Models
* **Concept**: Converting text into vector lists (floating point numbers) to measure relatedness.
* **Code Topic**: **Hugging Face (Local vs Online)**
```python
# Online (API-based, lightweight)
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Local (Downloads model, runs offline)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
```

* **Code Topic**: **Cosine Similarity (Manual Implementation)**
```python
def calculate_score(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (mag1 * mag2)
```
