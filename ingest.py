import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

def load_documents(data_dir):
    docs = []
    for file_path in Path(data_dir).glob("*"):
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
        elif file_path.suffix.lower() == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY not set.")

    documents = load_documents(DATA_DIR)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print("Vectorstore created successfully")

if __name__ == "__main__":
    main()
