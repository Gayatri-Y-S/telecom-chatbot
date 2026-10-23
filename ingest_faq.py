"""
Ingests data/faq.csv into the 'faq' Chroma collection.
Run once (or whenever the CSV changes): python ingest_faq.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import pandas as pd
#It imports LangChain's Document class, which stores text content along with metadata.
#Document(page_content=content, metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])})
from langchain_core.documents import Document
#It imports Chroma, which connects LangChain to the ChromaDB vector database.
from langchain_chroma import Chroma
#: It imports the class that converts text into numerical vectors (embeddings) using a Hugging Face model.
from langchain_huggingface import HuggingFaceEmbeddings

#Specifies the folder where ChromaDB stores the vector database on your computer.
#ChromaDB is the tool; the vector database is what it manages.
CHROMA_DIR = "chroma_store"
#Example: Think of ChromaDB as a library and faq as a shelf containing all your FAQ data.
#This tells ChromaDB to store the FAQ embeddings in the faq collection.
COLLECTION  = "faq"
#Use: Specifies the path (location) of your FAQ CSV file. - os.path.join() combines folder and file names into a path. - "data" is the folder. - "faq.csv" is the file inside that folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "faq.csv")
#Use: Specifies which embedding model your chatbot uses to convert text into numerical vectors. - sentence-transformers → the model library/organization. - all-MiniLM-L6-v2 → the name of the embedding model.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    #print("###############df",df)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])}, 
        ))
    #print("##############docs",docs)
    return docs


def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)
    print(f"  {len(docs)} FAQ entries loaded.")

    print("Initialising embedding model...")
    #Load the all-MiniLM-L6-v2 model and use it to convert my FAQ questions/answers into vectors.
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()
