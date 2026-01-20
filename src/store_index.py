


from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medbot"

# Load and process documents
extracted_data = load_pdf_file(data="data/")
texts = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Store documents in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=index_name,
)
