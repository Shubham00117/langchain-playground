# Notes Creation Prompt

When the user provides code files or a folder for a new day's notes, follow this exact pattern:

---

## Structure

```
# Day X Notes

---

# Module N: Module Name

---

## N. Topic Title

**Concept:** ...

\```python
...
\```

> one-liner summary

---
```

---

## Pattern Rules

### 1. Title
- Start with `# Day X Notes`
- Ask the user which day number if not mentioned.

### 2. Modules
- Group related topics under `# Module N: Module Name`
- Use `---` before and after each module heading.
- Examples: Prompts, Chains, Memory, Structured Output, Output Parsers, etc.

### 3. Numbered Topics
- Format: `## N. Topic Title`
- Numbering is **continuous across all modules** (1, 2, 3... not restarting per module).
- Use `---` between every topic.

### 4. Concept Definition
- Starts with `**Concept:**`
- Written in **simple, everyday language** — like explaining to a friend.
- No technical jargon. No class/method names in the definition.
- Focus on **what it does and why you'd use it**, not how the code works internally.
- Keep it 1–2 lines max.

### 5. Code Snippet
- Only include lines that demonstrate the core concept.
- **Remove** all boilerplate: env loading, path logic, `load_dotenv`, `os.path`, `__name__` blocks.
- **Keep imports** only if the import itself shows the concept (e.g., `from langchain_core.prompts import load_prompt`).
- **Remove** imports that aren't central to the concept (e.g., `os`, `dotenv`).

### 6. Comments in Code
- Write comments as **title lines above the code block** they describe — not inline.
- Only add a comment where a **unique or important concept** appears that needs explaining.
- Comments should be **short and clear** — like a label for what the next few lines do.
- Do NOT comment on obvious lines.
- Example pattern:
  ```python
  # load prompt from external JSON file
  prompt = load_prompt(json_path)
  
  # bind schema to model — now it returns Review objects
  structured_llm = llm.with_structured_output(Review)
  ```

### 7. One-Liner Summary
- End each topic with a `>` blockquote.
- Single line showing the flow using `→` arrows.
- Uses backtick-wrapped code names.
- Example: `` > `load_prompt(path)` → reads JSON → returns `PromptTemplate` → chain with any LLM. ``

---

## Steps

1. Read all code files the user provides.
2. Identify unique concepts from each file.
3. Group them into logical modules.
4. For each topic:
   - Write a simple concept definition (no jargon).
   - Extract minimal code with only concept-relevant imports.
   - Add short title-style comments only where a unique concept appears.
   - Write a one-liner arrow summary.
5. Save to `/Users/shubham_infinity/Desktop/test1/LangChain_Project/notes/` (append or create new file as user prefers).
