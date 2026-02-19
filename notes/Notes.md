# Day 2 Notes

---

# Module 1: Prompts

---

## 1. Load Prompt from JSON File

**Concept:** Instead of writing your prompt inside the code, you save it in a `.json` file. Then use `load_prompt()` to load it. This way you can reuse the same prompt anywhere without copy-pasting.

```python
# loads prompt template from a JSON file
from langchain_core.prompts import load_prompt

# load prompt from external JSON file
json_path = os.path.join(current_dir, "prompt.json")
prompt = load_prompt(json_path)

# create model and chain
model = ChatGroq(model_name="llama-3.3-70b-versatile")
chain = prompt | model
response = chain.invoke({"paper": "BERT", "style": "Beginner", "length": "Short"})
print(response.content)
```

> `load_prompt(path)` → reads JSON → returns `PromptTemplate` → chain with any LLM.

---

## 2. Messages (SystemMessage & HumanMessage)

**Concept:** Instead of sending a plain string, you send a **list of messages** to the LLM. Each message has a role — `SystemMessage` tells the AI how to behave, `HumanMessage` is what the user says, and `AIMessage` is what the AI replies.

```python
# import message types for role-based conversation
from langchain_core.messages import SystemMessage, HumanMessage

# message list with roles
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is langchain?")
]

# send list to model and save AI reply back
response = model.invoke(messages)
messages.append(response)
```

> `SystemMessage` → AI role | `HumanMessage` → user query | `AIMessage` → model reply → all in one list.

---

## 3. Chatbot with Conversation History

**Concept:** To make the AI remember what was said before, keep all messages in a **list**. Every time the user asks something, add it to the list, send the whole list to the AI, and save the AI's reply back. This way the AI always sees the full conversation.

```python
# message types for user input and AI response
from langchain_core.messages import HumanMessage, AIMessage

# list acts as memory
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    # add user msg → send full history → save AI reply
    chat_history.append(HumanMessage(content=user_input))
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
```

> Loop → append `HumanMessage` → invoke full list → append `AIMessage` → the list **is** the memory.

---

## 4. ChatPromptTemplate

**Concept:** A quick way to create prompts with **roles and blanks**. You write `("system", "...")` and `("human", "...")` with `{placeholders}`, then fill in the values later using `.invoke()`. Cleaner than creating message objects by hand.

```python
# template with roles and placeholders
from langchain_core.prompts import ChatPromptTemplate

# define prompt using (role, template) tuples
template = ChatPromptTemplate.from_messages([
    ("system", "Translate {input_language} to {output_language}."),
    ("human", "{text}"),
])

# fill placeholders at runtime
prompt_value = template.invoke({
    "input_language": "English",
    "output_language": "French",
    "text": "I love LangChain!"
})

response = model.invoke(prompt_value)
```

> `("role", "{var}")` tuples → `.invoke(dict)` fills variables → returns formatted prompt.

---

## 5. MessagesPlaceholder

**Concept:** Adds an **empty slot** in your prompt template where you can plug in old chat messages later. You can load past conversation from a file and pass it in when calling the chain — the AI will see the full history.

```python
# MessagesPlaceholder creates a slot for injecting chat history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful support agent"),
    # empty slot — past messages will be injected here
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

# load old conversation from file
with open("history.txt") as f:
    chat_history = f.readlines()

# history gets injected into the placeholder at invoke time
chain = chat_template | model
response = chain.invoke({"chat_history": chat_history, "query": "Where is my refund?"})
```

> `MessagesPlaceholder("chat_history")` → slot in template → pass history at invoke time.

---

# Module 2: Structured Output

---

## 6. Structured Output — Pydantic

**Concept:** You tell the AI **exactly what shape** the answer should be using a Pydantic class (like a form with fields). The AI fills in the fields and you get back a proper Python object — not random text.

```python
# BaseModel defines the output shape, Field adds descriptions
from pydantic import BaseModel, Field

# define output shape as a class
class Review(BaseModel):
    summary: str = Field(description="Brief summary")
    pros: list[str] = Field(description="Positive points")
    sentiment: str = Field(description="Overall tone")

# bind schema to model — now it returns Review objects
structured_llm = llm.with_structured_output(Review)
result = structured_llm.invoke("review text...")
# convert Pydantic object to dict
print(result.model_dump())
```

> `BaseModel` + `Field(description=...)` → `.with_structured_output()` → validated Pydantic object.

---

## 7. Structured Output — TypedDict

**Concept:** Same idea as Pydantic but **simpler** — you define the shape using `TypedDict` instead. No validation or type checking, you just get back a normal Python **dictionary**.

```python
# TypedDict for schema, Annotated for field descriptions
from typing import TypedDict, Annotated

# Annotated adds description for each field
class Review(TypedDict):
    summary: Annotated[str, "Brief summary"]
    pros: Annotated[list[str], "Positive points"]
    sentiment: Annotated[str, "Overall tone"]

# same method — but returns a plain dict instead of object
structured_llm = llm.with_structured_output(Review)
result = structured_llm.invoke("review text...")
```

> `TypedDict` + `Annotated` → no validation → returns plain dict.

---

## 8. Structured Output — JSON Schema

**Concept:** Instead of writing a Python class, you define the output shape as a **plain JSON dictionary**. Handy when the schema is already in a JSON file or comes from somewhere else.

```python
# define schema as raw dict — no Python class needed
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Brief summary"},
        "sentiment": {"type": "string", "description": "Overall tone"}
    },
    "required": ["summary", "sentiment"]
}

# pass dict directly as schema
structured_llm = llm.with_structured_output(json_schema)
result = structured_llm.invoke("review text...")
```

> JSON dict → `.with_structured_output()` → no classes needed → returns plain dict.



---

# Module 3: Output Parsers

---

## 9. StrOutputParser

**Concept:** By default, an LLM returns a complex message object. `StrOutputParser` extracts just the text from that response, giving you a clean string. This is useful when you only care about the answer text, not the metadata.

```python
# extracts plain string from AIMessage
from langchain_core.output_parsers import StrOutputParser

# parser at the end strips AIMessage wrapper → gives plain string
chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": "the sun"})
```

> `StrOutputParser()` → extracts `.content` → returns clean string.

---

## 10. StrOutputParser — Chain Two Prompts

**Concept:** You can use the output of one chain as the input for another. `StrOutputParser` makes this easy by converting the first AI's response into a plain string that the second prompt can easily read.

```python
parser = StrOutputParser()

# chain 1 generates detailed answer
chain1 = prompt1 | model | parser
# chain 2 summarizes it
chain2 = prompt2 | model | parser

# feed chain1 output into chain2
result = chain1.invoke({"topic": "Quantum Computing"})
summary = chain2.invoke({"input": result})
```

> Chain1 → string output → feed into Chain2 → sequential prompt chaining.

---

## 11. JsonOutputParser

**Concept:** Automatically converts the AI's response from a JSON string into a Python dictionary. It also provides instructions to tell the AI exactly how to format its response as JSON.

```python
# parses AI's JSON response into a Python dict
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# get_format_instructions() injects JSON format rules into prompt
template = PromptTemplate(
    template="Give 5 facts about {topic}\n{format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# parser converts JSON string response → Python dict
chain = template | model | parser
result = chain.invoke({"topic": "black hole"})
```

> `JsonOutputParser()` → `get_format_instructions()` → LLM replies JSON → parsed to dict.

---

## 12. PydanticOutputParser

**Concept:** The most advanced parser. It tells the AI exactly what format to use (based on a class you define) and then validates that the AI's response is correct before turning it into a Python object.

```python
# validates and converts AI response into a Pydantic object
from langchain_core.output_parsers import PydanticOutputParser
# BaseModel defines schema, Field adds descriptions
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Name")
    age: int = Field(description="Age")
    city: str = Field(description="City")

# parser knows the expected shape
parser = PydanticOutputParser(pydantic_object=Person)

# get_format_instructions() tells AI exactly which fields to return
template = PromptTemplate(
    template="Extract person info: {query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# parser validates response and converts to Person object
chain = template | model | parser
result = chain.invoke({"query": "John is 25 and lives in New York"})
```

> `PydanticOutputParser(pydantic_object=Model)` → injects instructions → parses + validates → typed object.

# Day 3 Notes

---

# Module 4: Chains

---

## 13. Basic Chain

**Concept:** A chain is just a series of steps where data flows from one component to another. You connect a prompt, a model, and a parser together using the pipe (`|`) operator to build a complete workflow.

```python
parser = StrOutputParser()
# simple chain connecting prompt, model, and parser
chain = prompt | model | parser

# invoke starts the flow
result = chain.invoke({"topic": "AI"})
```

> `prompt` → `model` → `parser` → returns clean response string.

---

## 14. Sequential Chain

**Concept:** Used when you want to perform tasks one after another, where the output of the first task is given to the second task. For example, generating a long report first and then creating a summary of that report.

```python
parser = StrOutputParser()
# chain 1 generates a long report
chain1 = prompt1 | model | parser

# chain 2 takes string and summarizes it
# lambda maps the string to a dictionary key that prompt2 expects
full_chain = chain1 | (lambda x: {"text": x}) | prompt2 | model | parser

result = full_chain.invoke({"topic": "Generative AI"})
```

> `chain1` output → mapped to dict → fed into `chain2` → sequential multi-step task.

---

## 15. Parallel Chain

**Concept:** Allows you to run multiple tasks at the exact same time using the same input. Useful when you want to generate different types of content simultaneously, like study notes AND a quiz from the same text.

```python
from langchain_core.runnables import RunnableParallel

parser = StrOutputParser()
# define two chains to run at once
parallel_chain = RunnableParallel({
    "notes": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

# result will be a dictionary containing both 'notes' and 'quiz'
result = parallel_chain.invoke({"text": data})

#create merge chain
merge_chain = prompt3 | model1 | parser

#combine into final chain
chain = parallel_chain | merge_chain

result = chain.invoke({"text": data})
```

> `RunnableParallel({key: chain})` → runs all chains in parallel -> merge logic → returns dict of results.

---

## 16. Conditional Chain (Branching)

**Concept:** Acts like an "if-else" statement for your AI. The chain looks at the input (or output of a previous step) and decides which path to take based on a condition you set.

```python
from langchain_core.runnables import RunnableBranch

# define logic: if positive do X, if negative do Y
branch = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_chain),
    (lambda x: x["sentiment"] == "negative", negative_chain),
    default_chain
)

# full chain first classifies then branches
full_chain = classification_chain | branch
```

> `RunnableBranch((condition, chain), default)` → checks condition → triggers specific branch.

---

# Module 5: Runnable

---

## 17. The Runnable Concept

**Concept:** This is the common "rulebook" that all LangChain components follow. Because they all implement the `invoke()` method, they can be easily "piped" together like building blocks.

```python
# Think of .invoke() as the universal "Start" button for every component
# Data flows: Input -> [ PROMPT ] -> [ MODEL ] -> [ PARSER ] -> Output

# 1. Manual Flow (Each block has its own .invoke())
val1 = prompt.invoke({"topic": "AI"})
val2 = model.invoke(val1)
final = parser.invoke(val2)

# 2. Piped Flow (All blocks unified into one .invoke())
chain = prompt | model | parser
final = chain.invoke({"topic": "AI"})
```

> `Runnable` interface → standard `.invoke()` method → allows all components to connect.

---

## 18. LCEL (The Pipe Operator)

**Concept:** The "pipe" operator (`|`) is the heart of LangChain Expression Language. It provides a visual and intuitive way to chain components together, allowing data to flow seamlessly from one step to the next while automatically handling the underlying logic of a `RunnableSequence`.

```python
# data flows from left to right
# prompt logic -> model processing -> parser output extraction
chain = prompt | model | parser
```

> `|` operator → connects runnables → creates `RunnableSequence` automatically.

---

## 19. RunnableSequence

**Concept:** `RunnableSequence` is the underlying container that manages a series of steps in a chain. It can be created explicitly using the `RunnableSequence` class or automatically by using the pipe operator (`|`). It ensures that data flows through each component in the exact order they are defined.

```python
from langchain_core.runnables import RunnableSequence

# grouping existing chains and components explicitly
final_chain = RunnableSequence(
    initial_chain,
    summary_prompt,
    model,
    parser
)

# same chain but using pipe operator
final_chain = (
    initial_chain
    | summary_prompt
    | model
    | parser
)
```

> `RunnableSequence(step1, step2, ...)` → explicit class construction of a chain.

---

## 20. RunnableParallel

**Concept:** This is the tool that makes multi-tasking possible. By wrapping your chains in a dictionary format inside `RunnableParallel`, they all get executed at the same time.

```python
from langchain_core.runnables import RunnableParallel

# dictionary keys become the labels for the parallel outputs
chain = RunnableParallel(
    tweet = twitter_chain,
    linkedin = linkedin_chain
)
```

> `RunnableParallel(name=chain)` → name-based parallel execution → returns dictionary.

---

## 21. RunnablePassthrough

**Concept:** `RunnablePassthrough` is used to capture the output of a previous step and assign it directly to a dictionary key without changing it. This allows you to "branch" the flow: one branch preserves the original result in a specific key, while other branches use that same data as input for further processing.

```python
from langchain_core.runnables import RunnablePassthrough

# 1. Define the initial step (e.g., generate a joke)
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# 2. Create a parallel block to process the output from step 1
# This keeps the original output while also performing a second task
parallel_chain = RunnableParallel({
    "original_joke": RunnablePassthrough(), # Passes the joke along unchanged
    "explanation": explanation_prompt | model | parser # Processes the joke for an explanation
})

# 3. Combine the chains
final_chain = joke_gen_chain | parallel_chain | parser
```

> `RunnablePassthrough()` → takes input data → returns it exactly as it was received.

---

## 22. RunnableLambda

**Concept:** Allows you to turn any normal Python function into a LangChain component. If you have a custom bit of logic (like counting words or cleaning text) that isn't a prompt or a model, you wrap it in a lambda.

```python
from langchain_core.runnables import RunnableLambda

# Custom function to be used with RunnableLambda
def word_count(text):
    return len(text.split())

# 2. Parallel Chain with RunnableLambda
# RunnableLambda allows you to run custom Python functions within a chain
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

# 3. Final Sequence
final_chain = joke_gen_chain | parallel_chain | parser
```

> `RunnableLambda(function)` → converts Python function → into a chainable component.

---

## 23. RunnableBranch

**Concept:** The official way to handle routing in LangChain. You provide a list of (condition, runnable) pairs, and it executes the first one where the condition is true.

```python
from langchain_core.runnables import RunnableBranch

# routing based on text length
branch = RunnableBranch(
    (lambda x: len(x) > 500, summary_chain),
    RunnablePassthrough() # default: if short, do nothing
)
```

> `RunnableBranch` → list of `(if, then)` → first true condition wins.

---

# Day 4 Notes

---

# Module 6: Document Loaders

---

## 24. TextLoader

**Concept:** The simplest loader that reads a standard `.txt` file and loads it as a Document. It treats the entire file content as one single document.

```python
from langchain_community.document_loaders import TextLoader

# load the text file
loader = TextLoader("example.txt")
docs = loader.load()

# access content
print(docs[0].page_content)
```

> `TextLoader(path)` → loads `.txt` → returns list with 1 Document.

---

## 25. CSVLoader

**Concept:** Loads a `.csv` file where **each row** becomes a separate Document. Ideally used when you want to treat every row (e.g., a product, a tweet) as an individual record.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# load csv where each row becomes a document
loader = CSVLoader(file_path="data.csv")
data = loader.load()

# Combine all rows into a single text block for the LLM
context = "\n".join([doc.page_content for doc in data])

print(len(data)) # equals number of rows
```

> `CSVLoader(path)` → iterates rows → 1 Document per row.

---

## 26. PyPDFLoader

**Concept:** Extracts text from a PDF file. It creates one Document **per page**, preserving the page number in the metadata. Best for simple PDFs.

```python
from langchain_community.document_loaders import PyPDFLoader

# load pdf, splits by page
loader = PyPDFLoader("paper.pdf")
pages = loader.load()

# access page 1 content
print(pages[0].page_content)
print(pages[0].metadata) # {'source': '...', 'page': 0}
```

> `PyPDFLoader(path)` → reads PDF → 1 Document per page.

---

## 27. WebBaseLoader

**Concept:** Scrapes text from a webpage URL. It fetches the HTML, cleans it (removes tags), and loads the readable text as a Document.

```python
from langchain_community.document_loaders import WebBaseLoader

# scrape a website
loader = WebBaseLoader("https://example.com")
docs = loader.load()

print(docs[0].page_content)
```

> `WebBaseLoader(url)` → fetches HTML → extracts text → returns Document.

---

## 28. DirectoryLoader

**Concept:** Loads **all files** in a folder matching a pattern (like `*.pdf`). It uses a specific loader (like `PyPDFLoader`) to process each file it finds.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# load all PDFs in 'books' folder
loader = DirectoryLoader(
    path='books',        # folder where files are stored
    glob='*.pdf',       # file pattern to match (e.g., all PDFs)
    loader_cls=PyPDFLoader # specific loader to use for each file found
)
docs = loader.load()
```

**Common Glob Patterns:**
- `*.pdf` → Only PDFs in the main folder.
- `**/*.pdf` → PDFs in the main folder **and all subfolders** (recursive).
- `**/*.txt` → All Text files recursively.
- `*.*` → Every file in the main folder.

> `DirectoryLoader(path, glob)` → finds files → delegates to `loader_cls` → aggregated docs.

---

## 29. Lazy Loading

**Concept:** Instead of loading ALL documents into memory at once (which crashes RAM with big datasets), `lazy_load()` yields documents **one by one**. You process the current document and discard it before moving to the next.

```python
# standard load - high memory usage
# docs = loader.load()

# DirectoryLoader supports lazy_load() which returns a generator
# lazy load - efficient memory usage
docs_generator = loader.lazy_load()

for doc in docs_generator:
    print(doc.page_content)
    # process doc here, then it frees memory
```

> `lazy_load()` → returns generator → yields 1 doc at a time → saves RAM.

---

# Module 7: Text Splitters

---

## 30. CharacterTextSplitter

**Concept:** Splits text purely based on a **character count** (e.g., every 100 characters). It’s the simplest method but can break words or sentences in the middle, losing context.

**What is `chunk_overlap`?**
It keeps a small piece of text from the end of one chunk and repeats it at the start of the next. This ensures that the context (like a sentence or phrase) isn't completely cut off and helps the AI understand the connection between pieces.

```python
from langchain_text_splitters import CharacterTextSplitter

# simple split with overlap
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20, # repeats 20 characters from previous chunk
    separator=""
)
docs = splitter.split_documents(original_docs)
```

> `CharacterTextSplitter` → strict length split + `chunk_overlap` for context preservation.

---

## 31. RecursiveCharacterTextSplitter

**Concept:** The **standard go-to splitter**. It tries to split at smart points—first by paragraphs `\n\n`, then newlines `\n`, then spaces—to keep related text together. It only cuts words if strictly necessary to fit the chunk size.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# splits intelligently: \n\n -> \n -> space
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = splitter.split_documents(original_docs)
```

> `RecursiveCharacterTextSplitter` → tries smart separators → keeps context intact → best default.

---

## 32. Split by Language (Python/Markdown)

**Concept:** A version of RecursiveSplitter giving it **specific rules** for code or markdown. For Python, it knows to split at `class` or `def`; for Markdown, at `# headers`. This keeps logical blocks (like a whole function) intact.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# split python code keeping classes/functions together
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=200
)
# create_documents expects a list because it can process multiple text sources at once
docs = python_splitter.create_documents([python_code])
```

> `.from_language(Language.X)` → respects syntax boundaries (functions, classes) → better for code.

---

## 33. SemanticChunker

**Concept:** Instead of purely splitting by size, this uses **embeddings** to understand meaning. It splits text only when the **topic changes** drastically (e.g., shifting from "Roman Empire" to "Organic Chemistry"). This creates "meaning-aware" chunks.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# splits when semantic meaning shifts
splitter = SemanticChunker(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    # splits when meaning shifts significantly from the average
    breakpoint_threshold_type="standard_deviation",
    # Robust threshold for large, detailed paragraphs
    breakpoint_threshold_amount=1.2 
)
# processes raw text and returns a list of semantic Documents
chunks = splitter.create_documents([text]) 
```

> `SemanticChunker` + Embeddings → detects topic shifts → splits by meaning (not size).
