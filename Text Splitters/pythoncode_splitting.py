from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 1. Define the Python code as a text variable
python_code = """
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        return f"Student: {self.name}, Age: {self.age}"

# Creating an object of the Student class
student_obj = Student("Shubham", 25)
print(student_obj.display_info())
"""

# 2. Use RecursiveCharacterTextSplitter for Python
# We use from_language to handle Python-specific syntax like 'class', 'def', etc.
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=250, # Increased to keep the class in one part and the object in the second
    chunk_overlap=0
)

# 3. Perform the splitting
python_docs = python_splitter.create_documents([python_code])

# 4. Output the results
print(f"Total chunks created: {len(python_docs)}")

print("\n--- PART 1 (CLASS) ---")
print(python_docs[0].page_content)

print("\n--- PART 2 (OBJECT) ---")
print(python_docs[1].page_content)
