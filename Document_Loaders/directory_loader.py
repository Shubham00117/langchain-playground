from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

# 1. Print total length (Total pages across all PDFs)
# 1. Print total pages (Total documents in the list)
print(f"Total Pages Loaded: {len(docs)}")

# 2. Print total unique PDF files
unique_files = set(doc.metadata['source'] for doc in docs)
print(f"Total PDF Files: {len(unique_files)}")

# 3. Print metadata of the first page
print("\n--- METADATA ---")
print(docs[1].metadata)

# 4. Print the content of the first page for the first three PDF files using their indices
print("\n--- content of first pdf of first page ---")
print(docs[0].page_content)
print("\n--- content of second pdf of first page ---")
print(docs[61].page_content)
print("\n--- content of third pdf of first page ---")
print(docs[122].page_content)
    
 