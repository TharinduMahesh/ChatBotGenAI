# from langchain.document_loaders import PyPDFLoader,DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings

# def load_pdf_file(data):
#     loader = DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
#     documents = loader.load()
#     return documents

# def text_split(extracted_data):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     texts = text_splitter.split_documents(extracted_data)
#     return texts

# def download_hugging_face_embeddings():
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     return embeddings

# src/helper.py
# src/helper.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Updated
from langchain_huggingface import HuggingFaceEmbeddings # Updated

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(extracted_data)
    return texts

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings