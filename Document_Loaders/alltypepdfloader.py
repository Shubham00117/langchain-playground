"""
DEMONSTRATION OF DIFFERENT PDF LOADERS IN LANGCHAIN
Based on common use cases for better data extraction.
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader
)

# 1. PyPDFLoader
# USE CASE: Simple, clean PDFs.
# Logic: Uses the 'pypdf' library under the hood. It is fast but struggles 
# with complex layouts, tables, or scanned images.
def demo_pypdf():
    loader = PyPDFLoader("sample.pdf")
    pages = loader.load() # Returns a list of Document objects (one per page)
    print("PyPDFLoader: Simple and fast strategy.")

# 2. PDFPlumberLoader
# USE CASE: PDFs with Tables or complex multi-column layouts.
# Logic: Detailed extraction of each text character, rectangle, and line. 
# Best for preserving table structures.
def demo_pdfplumber():
    loader = PDFPlumberLoader("table_data.pdf")
    data = loader.load()
    print("PDFPlumberLoader: Best for tables and columns.")

# 3. PyMuPDFLoader (fitz)
# USE CASE: When you need high speed AND layout/image data.
# Logic: One of the fastest parsing strategies. It extracts a lot of 
# technical metadata about the document layout.
def demo_pymupdf():
    loader = PyMuPDFLoader("complex_layout.pdf")
    data = loader.load()
    print("PyMuPDFLoader: Extremely fast and good for layout data.")

# 4. UnstructuredPDFLoader
# USE CASE: Scanned/Image PDFs or when you want the "Best" structure extraction.
# Logic: Heavy-duty loader. Can use 'OCR' (Tesseract) to read text from images.
# It labels elements (Title, NarrativeText, ListItems, etc.).
def demo_unstructured():
    # mode="elements" captures the structure (Titles, Tables, etc.)
    loader = UnstructuredPDFLoader("scanned_doc.pdf", mode="elements")
    data = loader.load()
    print("UnstructuredPDFLoader: Best for structure and scanned images.")

# 5. AmazonTextractPDFLoader (Cloud Based)
# USE CASE: Enterprise-grade OCR for scanned documents.
# Note: Requires AWS credentials and an active AWS account.
# Best for extremely messy scanned forms.
# from langchain_community.document_loaders import AmazonTextractPDFLoader

print("File created to demonstrate the recommended loaders based on usage requirements.")
