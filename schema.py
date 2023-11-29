import keyword
from typing import Optional
from fastapi import Body
from pydantic import BaseModel

class TrainingRequest(BaseModel):
    charter: str = Body(None, description="Charter that needs to train")
    topic: Optional[str] = Body("", description="Topic")
    subject: Optional[str] = Body("", "Subject")

class BatchTrainingRequest(BaseModel):
    charter_list: list = Body(None, description="List of charters that need to train")
    topic: Optional[list] = Body(None, description="Topic List")
    subject: Optional[list]  = Body(None, "Subject List")


class ChatRequest(BaseModel):
    messages: list = Body([], description="Chat History")

class SearchRequest(BaseModel):
    topic: str = Body("", description="Topic")
    subject: str = Body("", "Subject")
    keyword: str = Body("", "Keyword")