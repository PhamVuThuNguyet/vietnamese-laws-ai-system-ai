"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""

from llm.base_model.langchain_openai import LangchainOpenAI

async def summarize(question: str) -> str:
    chain = LangchainOpenAI(question)
    response = chain.summarize_question(question)
    return response