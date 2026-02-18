from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# 1. Robust Industry Standard Path Handling
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "a_study_in_scarlet.pdf"

# 2. Load the PDF
loader = PyPDFLoader(str(file_path))
docs = loader.load()

# 3. Initialize the Splitter (Length-based)
# As per screenshot logic: chunk_size=100, chunk_overlap=0, separator=''
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

# 4. Perform the splitting
# We can pass the list of documents directly to split_documents
final_docs = splitter.split_documents(docs)

# 5. Output Results
print(f"Total pages in PDF: {len(docs)}")
print(f"Total chunks created: {len(final_docs)}")
print(f"Total content of  chunk: {final_docs}")

print("\n--- FIRST CHUNK CONTENT ---")
print(final_docs[0].page_content)

print("\n--- SECOND CHUNK CONTENT ---")
print(final_docs[1].page_content)

print("\n--- five CHUNK CONTENT ---")
print(final_docs[4].page_content)



print("\n--- METADATA OF FIRST CHUNK ---")
print(final_docs[0].metadata)
