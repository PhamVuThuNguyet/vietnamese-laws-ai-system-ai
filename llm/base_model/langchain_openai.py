import json
import logging
import os
import re
from typing import Tuple
from uuid import UUID

import backoff
import openai
import yaml
from fastapi import HTTPException
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import ConversationalRetrievalChain
"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import MergerRetriever
from langchain.schema import HumanMessage
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

from core.constants import IngestDataConstants, LangChainOpenAIConstants
from llm.base_model.retrieval_chain import CustomConversationalRetrievalChain
from llm.data_loader.load_langchain_config import LangChainDataLoader
from llm.data_loader.vectorstore_retriever import CustomVectorStoreRetriever

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def openai_embedding_with_backoff():
    return OpenAIEmbeddings(chunk_size=IngestDataConstants.CHUNK_OVERLAP)


class LangchainOpenAI:
    """Langchain OpenAI"""

    def __init__(
        self,
        question,
        language: str = "Vietnamese",
        chat_history=None,
    ):
        self.output_parser = None
        self.is_chat_model, self.llm_cls, self.llm_model = self.load_llm_model()
        self.chat_history = chat_history

        self.data_loader = LangChainDataLoader()
        
        if language == None:
            self.lang = self._detect_language(question)
        else:
            self.lang = language

        vectorstore_folder_path = IngestDataConstants.VECTORSTORE_FOLDER

        self.vectorstore, self.vectorstore_retriever = self.get_langchain_retriever(
            vectorstore_folder_path=vectorstore_folder_path
        )

        self.data_loader.preprocessing_qa_prompt(
            language=self.lang,
            chat_history=self.chat_history,
        )

    def get_chain(self) -> ConversationalRetrievalChain:
        prompt_title = "qaPrompt"

        docs_chain = load_qa_chain(
            self.llm_model, prompt=self.data_loader.prompts[prompt_title]
        )
        return CustomConversationalRetrievalChain(
            retriever=self.vectorstore_retriever,
            combine_docs_chain=docs_chain,
            question_generator=LLMChain(
                llm=self.llm_model, prompt=self.data_loader.prompts["condensePrompt"]
            ),
            max_tokens_limit=8000,
            output_parser=self.output_parser,
            return_source_documents=True,
            return_generated_question=True,
        )

    @staticmethod
    def get_langchain_retriever(
        vectorstore_folder_path: str, vectorstore_search_kwargs: dict = None
    ) -> Tuple[VectorStore, MergerRetriever]:
        if vectorstore_search_kwargs is None:
            vectorstore_search_kwargs = {"k": 5, "score_threshold": 0.3}

        try:
            embeddings = openai_embedding_with_backoff()
            vectorstore = FAISS.load_local(
                folder_path=vectorstore_folder_path, embeddings=embeddings
            )
            vectorestore_retriever = CustomVectorStoreRetriever(
                vectorstore=vectorstore,
                search_type="similarity_score_threshold",
                search_kwargs=vectorstore_search_kwargs,
                metadata={"name": "help_center"},
            )

            final_vectorstore_retriever = MergerRetriever(
                retrievers=[vectorestore_retriever],
            )

            return vectorstore, final_vectorstore_retriever

        except Exception as e:
            raise HTTPException(
                status_code=500, detail="Error when loading vectorstore"
            )


    @staticmethod
    def load_llm_model():
        with open(
            os.path.join(
                LangChainOpenAIConstants.ROOT_PATH,
                f"configs/llms/{os.environ.get('PROMPT_VERSION', '031223')}.yaml",
            )
        ) as f:
            model_configs = yaml.safe_load(f)
        model_type = model_configs.pop("_type")
        llm_cls = LangChainOpenAIConstants.type_to_cls_dict_plus[model_type]
        llm_model = llm_cls(**model_configs)
        is_chat_model = isinstance(llm_model, ChatOpenAI)
        return is_chat_model, llm_cls, llm_model

    def _detect_language(self, question: str) -> str:
        try:
            language_detect_prompt = self.data_loader.prompts.get(
                "detectLanguagePrompt"
            ).template.format(
                question=question,
                chat_history="",
            )
            detected_language = (
                self.llm_model.generate(
                    [[HumanMessage(content=language_detect_prompt)]]
                )
                .generations[0][0]
                .text.strip()
            )
            regex = r"\%([^%]+)\%"
            language = re.findall(regex, detected_language)[-1]
        except Exception as e:
            language = "English"
        return language

    def summarize_question(self, question: str) -> str:
        try:
            summarize_prompt = self.data_loader.prompts.get(
                "summarizePrompt"
            ).template.format(question=question, lang=self.lang)

            summarization = (
                self.llm_model.generate([[HumanMessage(content=summarize_prompt)]])
                .generations[0][0]
                .text.strip()
            )
        except Exception as e:
            summarization = "New Conversation"
            raise HTTPException(
                status_code=500, detail="Error when loading summarization"
            )
        return summarization

    def _format_dict_list(self, dict_list: list[dict]):
        result = ""
        for item in dict_list:
            for category, info in item.items():
                result += f"{category.capitalize().replace('_', ' ')}: \n"
                result += json.dumps(info, indent=4).replace("{", "<").replace("}", ">")
                result += "\n\n"
        return result
