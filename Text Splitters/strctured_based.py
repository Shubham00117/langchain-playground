from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup paths
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "a_study_in_scarlet.pdf"

# 2. Load the document
print(f"Loading {file_path.name}...")
loader = PyPDFLoader(str(file_path))
docs = loader.load()

# 3. Initialize RecursiveCharacterTextSplitter
# This is favored in industry because it tries to keep paragraphs/sentences together
# by splitting on a list of characters: ["\n\n", "\n", " ", ""]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True # Good practice for tracking where the chunk came from
)

# 4. Perform the splitting
final_docs = splitter.split_documents(docs)

# 5. Output Results
print(f"\n--- RECURSIVE CHARACTER TEXT SPLITTER ---")
print(f"Total pages in PDF: {len(docs)}")
print(f"Total chunks created: {len(final_docs)}")

print("\n--- FIRST CHUNK ---")
print(final_docs[0].page_content)

print("\n--- SECOND CHUNK (Note the overlap) ---")
print(final_docs[1].page_content)

print("\n--- METADATA ---")
print(final_docs[0].metadata)
