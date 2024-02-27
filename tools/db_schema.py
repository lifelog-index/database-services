from typing import List, Optional
from pydantic import BaseModel

class ESFactModelV2(BaseModel):
    _id: str
    url: str
    claim: str
    content: str
    top_image: str

class ESFactModel(BaseModel):
    claim: str
    label: str
    label4level: str
    doc: str
    url: str
    domain: str
    domain_name: str
    publish_date: str
    keywords: list[str]
    authors: list
    search_engine: str
    lang: str
    score: float
    exact_match: bool


