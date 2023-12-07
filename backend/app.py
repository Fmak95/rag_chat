import sentencepiece as spm
from fastapi import FastAPI
from services.internal_classes import Request

from services.endpoints.invoke.service import Service


def load_tokenizer():
    tokenizer_1 = spm.SentencePieceProcessor(
        model_file="./models/mistral-7b/tokenizer.model"
    )
    return tokenizer_1


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/invoke")
def invoke(request: Request):
    response = Service().apply(request)
    return response
