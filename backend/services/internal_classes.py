from typing import Dict, List
from pydantic import BaseModel


class Request(BaseModel):
    messages: List[Dict]
