from typing import List
from pydantic import BaseModel


class Response(BaseModel):
    generated_token_ids: List[int]
    generated_text: str
