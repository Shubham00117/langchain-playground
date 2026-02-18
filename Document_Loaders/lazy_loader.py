from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Initialize the loader
# DirectoryLoader supports lazy_load() which returns a generator
loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

print("--- Lazy Loading Demonstration ---")
print("Instead of loader.load() (which brings everything into memory),")
print("we use loader.lazy_load() to process documents one by one.\n")

# loader.lazy_load() returns an iterator (generator)
docs_generator = loader.lazy_load()

print(f"Object Type: {type(docs_generator)}")

# Demonstrating lazy iteration
page_count = 0
for doc in docs_generator:
    page_count += 1
    
    # Print progress for every 50th page to show it's working
    if page_count % 50 == 0 or page_count == 1:
        source = doc.metadata.get('source', 'Unknown')
        page_num = doc.metadata.get('page', 0)
        print(f"Processed Page {page_count}: {source} (Page {page_num})")

print(f"\nTotal pages processed: {page_count}")
print("Concept: Documents were loaded, processed, and potentially cleared from memory one by one.")
