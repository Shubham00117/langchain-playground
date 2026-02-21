<div align="center">

# 🦜 LangChain Playground

**A comprehensive, structured learning repository for building Generative AI applications using LangChain.** 
From foundational prompt engineering to full-scale Retrieval-Augmented Generation (RAG) pipelines and AI Custom Agents.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Models-black?logo=openai&logoColor=white)](https://openai.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Chroma](https://img.shields.io/badge/Vector-Chroma%20|%20FAISS-teal)](https://docs.trychroma.com/)

*Based on the CampusX Generative AI Course Curriculum*

---

</div>

## 📖 Overview

LangChain Playground is an educational project designed to incrementally teach the core concepts of the LangChain ecosystem. It is divided into 13 logical modules, progressively scaling from simple API calls to building complete, autonomous AI Agents that can use tools.

### ✨ Key Features
* **Multi-Model Integration:** Seamlessly switch between OpenAI, Anthropic (Claude), Google (Gemini), and HuggingFace models.
* **Advanced RAG Pipelines:** Implement end-to-end RAG with smart text splitting, document loaders, vector stores (FAISS, Chroma), and specialized retrievers (MMR, Multi-Query).
* **LCEL (LangChain Expression Language):** Deep dive into Runnables (Sequence, Parallel, Branch, Lambda) for constructing modular chains.
* **Structured Output Parsing:** Ensure LLMs return predictable data formats (JSON, Pydantic objects) for robust application integration.
* **Tool Calling & Agents:** Create ReAct agents capable of making autonomous decisions and executing custom Python functions.
* **Beautiful HTML Study Notes:** Comprehensively documented theory across 88 structured topics (Module 1-13).

---

## 📂 Project Structure

```text
LangChain_Project/
├── 🤖 Models & Prompts
│   ├── LangChain_Models/       # Multi-provider Chat & Embedding Models
│   ├── PromptTemplate/         # Advanced Prompting (Few-shot, Prompt UI)
│   ├── OutputParsers/          # Str/JSON/Pydantic Output Parsers
│   └── StrcturedOutput/        # Enforced schema outputs (`.with_structured_output`)
│
├── ⛓️ Orchestration
│   ├── Chains/                 # Legacy sequential/parallel Chains
│   └── Runnable/               # Modern LCEL architecture (Runnables)
│
├── 📚 Retrieval Augmented Generation (RAG)
│   ├── Document_Loaders/       # Extracting text from PDFs, CSVs, Web, YouTube
│   ├── Text Splitters/         # Semantic, character, and code-aware chunking
│   ├── Vectors/                # FAISS and Chroma DB indexing
│   ├── Retrievers/             # Advanced context retrieval (MMR, Contextual Compression)
│   └── RAG/                    # The complete 4-stage RAG Pipeline execution
│
├── 🛠️ Autonomous Agents
│   └── Tool/                   # Tool definition, bindings, and AI Agent loops
│
└── 📝 Documentation
    └── notes/                  # Beautiful CSS-styled HTML study notes (Modules 1-13)
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.9+** installed on your machine.
- IDE of your choice (VS Code, PyCharm).
- API Keys for the models you intend to use (OpenAI, Gemini, HuggingFace, Anthropic).

### 2. Quick Start

Clone the repository and set up a virtual environment:

```bash
# Clone repo
git clone https://github.com/Shubham00117/langchain-playground.git
cd LangChain_Project

# Create and activate virtual environment
python -m venv myenv

# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate

# Install dependencies
pip install -r LangChain_Models/requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your necessary API keys:

```env
OPENAI_API_KEY=sk-your_openai_key
ANTHROPIC_API_KEY=sk-ant-your_claude_key
GOOGLE_API_KEY=AIza-your_gemini_key
HUGGINGFACEHUB_API_TOKEN=hf_your_huggingface_token
```

---

## 📚 Curriculum & Modules

The project code is mapped to 13 distinct learning modules. For detailed theoretical coverage of each concept, refer to the corresponding HTML file in the `/notes` directory.

| Module | Code Directory | Key Concepts Covered | Notes File |
|:---:|:---|:---|:---|
| **1** | `PromptTemplate/` | ChatPromptTemplate, MessagesPlaceholder, History | `Module_1_Prompts.html` |
| **2** | `StrcturedOutput/`| Structured outputs using Pydantic, TypedDict, JSON Schema | `Module_2_Structured_Output.html`|
| **3** | `OutputParsers/`  | PydanticOutputParser, JsonOutputParser, StrOutputParser| `Module_3_Output_Parsers.html` |
| **4** | `Chains/`         | Sequential, Parallel, and Conditional Chains | `Module_4_Chains.html` |
| **5** | `Runnable/`       | LCEL Pipe Operator, RunnableSequence, RunnableBranch | `Module_5_Runnable.html` |
| **6** | `Document_Loaders/`| PyPDF, CSV, WebBase, YouTube, Directory Loaders | `Module_6_Document_Loaders.html` |
| **7** | `Text Splitters/` | RecursiveCharacter, Language-specific, Semantic chunking | `Module_7_Text_Splitters.html` |
| **8** | `Vectors/` & `Retrievers/`| Embeddings, Vector Stores, Multi-Query, MMR Retrieval | `Module_8_Vector_Stores.html`, `Module_8_Retrievers.html` |
| **9** | `RAG/`            | Full RAG Pipeline: Indexing → Retrieval → Generation | `Module_9_RAG.html` |
| **10**| -                 | Complete RAG Implementation Example (YouTube Chatbot)| `Module_10_Youtube_Chatbot.html` |
| **11**| `Tool/`           | Built-in Tools, `@tool` decorator, Structured Custom Tools | `Module_11_Tools.html` |
| **12**| `Tool/`           | Tool Calling Mechanics, `bind_tools`, Tool Choice | `Module_12_Tool_Calling.html` |
| **13**| `Tool/`           | ReAct Agents, AgentExecutor, Multi-Tool autonomous loops | `Module_13_AI_Agent.html` |

---

## 📚 Study Notes (`/notes`)

This repository contains **13 beautifully crafted HTML study notes**. They serve as the theoretical backbone of the code structure here. 

To read them, simply open any `.html` file inside the `notes/` directory in your web browser. 

* **Topics 1-88:** The notes are sequentially numbered for a cohesive learning experience.
* **Visuals:** Features flowcharts, system diagrams, and color-coded definitions.
* **Code Snippets:** Includes inline Python snippets demonstrating core LangChain API calls.

---

## ⚙️ Tech Stack

This project heavily leverages the following Python libraries (defined in `requirements.txt`):
* `langchain` & `langchain-core` - Core framework orchestration.
* `langchain-openai`, `langchain-anthropic`, `langchain-google-genai` - LLM Provider Integrations.
* `langchain-huggingface`, `transformers` - Open-source models and local embeddings.
* `FAISS`, `chromadb` - Vector database solutions for semantic search.
* `pydantic` - Data validation and schema enforcement.

---
<div align="center">
<i>Structured Knowledge Base for LangChain Development — Built by Shubham | CampusX Generative AI Curriculum</i>
</div>
