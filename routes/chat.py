import logging

from langchain.callbacks import get_openai_callback
from langdetect import detect

from core.constants import ErrorChatMessageConstants
from core.utils import check_goodbye, check_hello, preprocess_suggestion_request
from llm.base_model.langchain_openai import LangchainOpenAI
from llm.data_loader.load_langchain_config import LangChainDataLoader
from schemas import ChatRequest
    
async def chat(request: ChatRequest) -> str:
    processed_request = preprocess_suggestion_request(request)

    question = processed_request.get("question")
    language = processed_request.get("language")

    chain = LangchainOpenAI(
        question=question,
        language=language if language else "Vietnamese",
    )

    try:
        with get_openai_callback() as cb:
            chat_history = processed_request.get("chat_history")
            
            qa_chain = chain.get_chain()
            response = await qa_chain.acall(
                    {
                        "question": question,
                        "chat_history": chat_history,
                        "dataset": "normal",
                    }
                )

    except Exception as e:
        logging.exception(e)
        lang = detect(question)
        answer = ErrorChatMessageConstants.VI if lang == "vi" else ErrorChatMessageConstants.EN
        return answer
    
    return response["answer"]