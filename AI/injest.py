from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

import gc
import uuid


load_dotenv()
embeddings = CohereEmbeddings(
    model='embed-english-v3.0'
)
CURRENT_DB = None
vector_store = None
def injest_pdf(pdf):
    global vector_store, CURRENT_DB
    vector_store = None
    gc.collect()

    loader = PyPDFLoader(pdf)
    docs = loader.load()
    
    #split the docs into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 100
    )
    chunks = splitter.split_documents(docs)
    
    #create vector and store into vector DB/Store
    CURRENT_DB = f"./pdfs/{uuid.uuid4()}"
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings,
        persist_directory=CURRENT_DB
        ) 
    print('db_created')
    
def load_db():
    global CURRENT_DB
    db = Chroma(
        persist_directory=CURRENT_DB,
        embedding_function=embeddings
    )
    return db