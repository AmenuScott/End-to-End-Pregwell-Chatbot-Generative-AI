# helper.py

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

#Extract Data from PDF files

def load_pdf_file(data):
    loader= DirectoryLoader( data,
                            glob="**/*.pdf",
                            loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents


#Split the data into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks=text_splitter.split_documents(extracted_data)

    return text_chunks


#download the embeddings from HuggingFace

def download_huggung_face_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2',
    )
    
    return embeddings

