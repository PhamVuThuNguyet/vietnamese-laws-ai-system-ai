import os
import backoff
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS

from typing import List, Text

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def openaiembedding_with_backoff():
    return OpenAIEmbeddings(chunk_size=500)

def ingest_plain_text(plain_text: Text):
    vectorstore_path = create_vectorstore()

    txt_path = vectorstore_path + "content.txt"
    with open(txt_path, "w", encoding="utf8") as fi:
        fi.write(plain_text)
    ingest_txt(txt_path)

def ingest_txt(txt_path: Text):
    vectorstore_path = create_vectorstore()

    loader = UnstructuredFileLoader(txt_path)
    raw_documents = loader.load()

    save_vectorstore(raw_documents, vectorstore_path)

def save_vectorstore(raw_documents: List, path: Text):
        text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo-16k",
            chunk_size=8000,
            chunk_overlap=100,
        )
        documents = text_splitter.split_documents(raw_documents)
        embeddings = openaiembedding_with_backoff()
        if os.path.exists(f"{path}/index.faiss"):
            vectorstore = FAISS.load_local(path, embeddings)
            vectorstore.add_documents(documents=documents)
        else:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vectorstore.save_local(path)

def create_vectorstore() -> Text:
        vectorstore_path = ""
        try:
            vectorstore_path = "/vectorstores"
            if not os.path.exists(vectorstore_path):
                os.makedirs(vectorstore_path)
        except Exception as e:
            pass
        finally:
            return vectorstore_path