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
