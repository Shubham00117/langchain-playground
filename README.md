# LangChain Project

A comprehensive environment for exploring and implementing Large Language Model (LLM) applications using **LangChain**. This project provides a structured setup to integrate with various AI providers and experiment with different LangChain components.

## 🚀 Key Features

*   **Multi-Model Integration**: Support for OpenAI, Anthropic (Claude), Google Gemini, and Hugging Face.
*   **Environment Management**: Structured for clean dependency management and API key security.
*   **Ready to Scale**: Modular structure designed for prototyping AI agents, chains, and RAG systems.

## 📁 Project Structure

```text
LangChain_Project/
├── README.md               # Project documentation
└── LangChain_Models/       # Core project directory
    ├── requirements.txt    # Python dependencies
    └── myenv/              # Virtual environment (local only)
```

## 🛠️ Getting Started

### 1. Set up Virtual Environment
Navigate to the `LangChain_Models` directory and create a virtual environment:

```bash
cd LangChain_Models
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
# myenv\Scripts\activate   # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the `LangChain_Models/` directory and add your API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

## 📚 Modules

### 🔮 Embedding Models & Semantic Search
Located in `LangChain_Models/3.EmbeddingModels/`, this module explores vector embeddings and similarity search using different providers.

*   **OpenAI Embeddings**:
    *   `1_Embedding_OpenAI_Single.py`: Generate single query embedding.
    *   `2_Embedding_OpenAI_Docs.py`: Generate embeddings for multiple documents.
*   **Hugging Face Embeddings**:
    *   `3_Embedding_HuggingFace_Local.py`: Run embedding models locally (offline).
    *   `4_Embedding_HuggingFace_Online.py`: Use Hugging Face Inference API (online).
    *   `5_Embedding_HuggingFace_Docs.py`: Batch processing for multiple documents.
*   **Semantic Search**:
    *   `6_Embedded_similarity.py`: Calculate cosine similarity to find the most relevant document for a query.

## 📦 Core Technologies

- **LangChain Ecosystem**: `langchain`, `langchain-core`
- **LLM Integrations**: OpenAI, Anthropic (Claude), Google Gemini, Hugging Face
- **Data Science Tools**: `numpy`, `scikit-learn`
- **Utilities**: `python-dotenv` for environment management

## 📝 Usage
Once setup is complete, you can start building:
1.  **Chains**: Sequence of LLM calls.
2.  **Agents**: LLMs that use tools.
3.  **RAG**: Retrieval Augmented Generation using vector stores.

---
*Created and maintained as a structured knowledge base for LangChain development.*
