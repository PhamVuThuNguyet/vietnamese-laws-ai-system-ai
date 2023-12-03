import keyword
from typing import Optional

from fastapi import Body
from pydantic import BaseModel


class TrainingRequest(BaseModel):
    charter: str = Body(None, description="Charter that needs to train")
    topic: Optional[str] = Body(None, description="Topic")
    subject: Optional[str] = Body(None, description="Subject")


class BatchTrainingRequest(BaseModel):
    charter_list: list = Body(None, description="List of charters that need to train")
    topic: Optional[list] = Body(None, description="Topic List")
    subject: Optional[list] = Body(None, description="Subject List")


class ChatRequest(BaseModel):
    """Request schema for QA. Chat history format:
    [
        {
            "role": "Role",
            "content": "content",    
        },
        {
            "role": "Role",
            "content": "content",    
        }
    ]"""
    messages: list = Body(None, description="List of chat history")
    language:  Optional[str] = Body(None, description="Language of expected response")


class SearchRequest(BaseModel):
    keyword: str = Body("", description="Keyword")
    topic: Optional[str] = Body(None, description="Topic")
    subject: Optional[str] = Body(None, description="Subject")
