from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 1. Define the Markdown content as a text variable
markdown_text = """
# LangChain Document Loaders

## File Loaders
- TextLoader: Loads plain text files.
- PyPDFLoader: Extracts text from PDFs.

## Web Loaders
- WebBaseLoader: Scrapes simple HTML pages.
- SeleniumURLLoader: Handles JavaScript-heavy sites.

### Summary
Document loaders are the foundation of many LLM applications, allowing models to process raw data from various sources.
"""

# 2. Use RecursiveCharacterTextSplitter for Markdown
# We use from_language to handle Markdown-specific syntax like headers and lists
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, 
    chunk_size=150, # Adjusted to create meaningful parts
    chunk_overlap=0
)

# 3. Perform the splitting
markdown_docs = markdown_splitter.create_documents([markdown_text])

# 4. Output the results
print(f"Total chunks created: {len(markdown_docs)}")

print("\n--- PART 1 ---")
print(markdown_docs[0].page_content)

print("\n--- PART 2 ---")
print(markdown_docs[1].page_content)

if len(markdown_docs) > 2:
    print("\n--- PART 3 ---")
    print(markdown_docs[2].page_content)
