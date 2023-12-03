import re
from typing import Tuple

from fastapi import HTTPException

from core.message_shortener import shorten_message
from schemas import ChatRequest


def preprocess_suggestion_request(request_body: ChatRequest):
    messages = request_body.messages
    language = request_body.language

    if len(messages):
        chat_history, question, previous_response = preprocess_chat_history(messages)
    else:
        raise HTTPException(status_code=400, detail="message is missing")

    return {
        "question": question,
        "chat_history": chat_history,
        "previous_response": previous_response,
        "language": language,
    }


def preprocess_chat_history(
    chat_history: list,
    max_words_each_message: int = 500,
    max_recent_chat_history: int = 4,
) -> Tuple[list, str, str]:
    new_chat_history = []
    question = ""
    previous_response = ""

    if len(chat_history) > max_recent_chat_history:
        chat_history = chat_history[-max_recent_chat_history:]

    if len(chat_history):
        current_item = None

        for item in chat_history:
            if current_item is None or item["role"] != current_item["role"]:
                if current_item is not None:
                    new_chat_history.append(current_item)
                current_item = {
                    "role": item["role"],
                    "content": item["content"].strip(),
                }
            else:
                current_item["content"] += "\n " + item["content"].strip()

        if current_item is not None:
            # Normalize the message string
            current_item["content"] = (
                current_item["content"].replace("\xa0", " ").replace("\\xa0", " ")
            )

            # Shorten message if it is too long
            content_word_len = len(re.findall(r"\w+", current_item["content"]))
            if content_word_len > max_words_each_message:
                current_item["content"] = shorten_message(
                    current_item["content"], max_words_each_message
                )

            new_chat_history.append(current_item)

    if new_chat_history[-1]["role"] == "user":
        question = new_chat_history[-1]["content"]
    else:
        if len(new_chat_history) == 1:
            question = new_chat_history[-1]["content"]
            del new_chat_history[-1]
        elif len(new_chat_history) > 1:
            previous_response = new_chat_history[-1]["content"]
            question = new_chat_history[-2]["content"]
            del new_chat_history[-1]

    chat_history = []

    if len(new_chat_history):
        index = 0
        start_role = new_chat_history[0]["role"]
        if start_role == "assistant":
            chat_history.append(("", new_chat_history[0]["content"]))
            index += 1

        while index < len(new_chat_history) - 1:
            if new_chat_history[index]["role"] == "user":
                chat_history.append(
                    (
                        new_chat_history[index]["content"],
                        new_chat_history[index + 1]["content"],
                    )
                )
            index += 2

    if question:
        question = list(question)
        question[0] = question[0].upper()
        question = "".join(question)

    return chat_history, question, previous_response