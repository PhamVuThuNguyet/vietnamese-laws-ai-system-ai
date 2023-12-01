import os
import backoff
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def openaiembedding_with_backoff():
    return OpenAIEmbeddings(chunk_size=500)

class ChatAgent():
    def __init__(self):
        self.embeddings = openaiembedding_with_backoff()
        if os.path.exists(f"/vectorstores/index.faiss"):
            self.vectorstore = FAISS.load_local("/vectorstores", self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3})
    
    def query_relevant_docs(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        return docs
    
    def chat(self, question: str, relevant_docs):
        if len(relevant_docs) > 0:
            final_doc = ""

            for i in range(relevant_docs):
                final_doc += f"\n{relevant_docs[i].page_content}"

            prompt = (f"Answer the question based only on the following knowledge base:\n"
                            f"{final_doc}"
                        f"Question: {question}")
            chat = ChatOpenAI(temperature=0)
            messages = [HumanMessage(content={prompt})]

            return chat(messages).content
