# 🦜 LangChain Project

A structured learning playground for building LLM applications with **LangChain** — from basic model calls to full RAG pipelines.

## 📁 Project Structure

```text
LangChain_Project/
├── LangChain_Models/       # Core models — OpenAI, Gemini, Claude, HuggingFace
├── PromptTemplate/         # Prompt engineering & template techniques
├── OutputParsers/          # Structured output — Pydantic, JSON, String parsers
├── StrcturedOutput/        # Structured output with Pydantic models
├── Chains/                 # Sequential & parallel LLM chains
├── Runnable/               # LCEL runnables — Sequence, Parallel, Lambda, Branch
├── Document_Loaders/       # PDF, CSV, Web, YouTube loaders
├── Text Splitters/         # Recursive, Character, Code-based splitting
├── Vectors/                # Embeddings, Vector stores, CRUD operations
├── Retrievers/             # Wikipedia, Vector Store, MMR, Multi-Query, Compression
├── RAG/                    # Full RAG pipeline — Indexing, Retrieval, Augmentation, Generation
└── notes/                  # HTML study notes for each module
```

## 🛠️ Setup

```bash
python -m venv myenv && source myenv/bin/activate
pip install -r LangChain_Models/requirements.txt
```

Add API keys to `.env`:
```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_token
```

## 📚 Modules

| Module | What's Inside |
|--------|--------------|
| **LangChain_Models** | Multi-model integration (OpenAI, Gemini, Claude, HuggingFace), embeddings & similarity search |
| **PromptTemplate** | ChatPromptTemplate, MessagesPlaceholder, few-shot prompts |
| **OutputParsers** | PydanticOutputParser, JSON/String parsers |
| **StrcturedOutput** | `.with_structured_output()` with Pydantic models |
| **Chains** | Sequential chains, LCEL pipe syntax |
| **Runnable** | RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch |
| **Document_Loaders** | PyPDFLoader, CSVLoader, WebBaseLoader, YouTubeLoader |
| **Text Splitters** | RecursiveCharacterTextSplitter, Language-based splitting |
| **Vectors** | OpenAI Embeddings, Chroma CRUD, FAISS |
| **Retrievers** | WikipediaRetriever, VectorStoreRetriever, MMR, MultiQueryRetriever, ContextualCompression |
| **RAG** | Complete 4-stage pipeline — Indexing, Retrieval, Augmentation, Generation |

## 📦 Core Stack

`langchain` · `langchain-openai` · `langchain-community` · `FAISS` · `Chroma` · `python-dotenv`

---
*Structured knowledge base for LangChain development — CampusX Generative AI Course*
