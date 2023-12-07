import os
import sys
from typing import List

sys.path.append(os.getcwd())

from llama_cpp import Llama
import torch
import sentencepiece as spm
from transformers import AutoTokenizer


def load_tokenizer():
    import sentencepiece.sentencepiece_model_pb2 as sp_proto_model

    tokenizer_1 = spm.SentencePieceProcessor(
        model_file="./models/mistral-7b/tokenizer.model"
    )

    # Test if tokenizer is working
    print(tokenizer_1.vocab_size())
    print(tokenizer_1.encode("<|im_end|>"))
    print(tokenizer_1.decode(32000))

    tokenizer_2 = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    return tokenizer_1


def generate_token_ids(messages: List[dict]):
    """
    Mistral LLM uses special encoding such that it adds special tokens to the prompt.

    <|im_start|>system
    You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
    <|im_start|>user
    Hello, who are you?<|im_end|>
    <|im_start|>assistant
    Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by a man named Teknium, who designed me to assist and support users with their needs and requests.<|im_end|>

    """

    templates: List[int] = []
    tokenizer = load_tokenizer()

    for message in messages:
        templates += (
            [32001]
            + tokenizer.encode_as_ids(message["role"])
            + tokenizer.encode_as_ids("\n")
            + tokenizer.encode_as_ids(message["content"])
            + [32000]
            + tokenizer.encode_as_ids("\n")
        )

    templates += [32001] + tokenizer.encode_as_ids(f"assistant\n")

    print("=" * 50)
    print(tokenizer.decode_ids(templates))
    return templates


def add_to_messages(messages: List[dict], prompt: str, role="user"):
    message = {"role": role, "content": prompt}
    messages.append(message)
    return


if __name__ == "__main__":
    # Load the model
    llm = Llama(
        model_path="./models/mistral-7b/openhermes-2.5-mistral-7b.Q3_K_M.gguf",
        verbose=False,
        main_gpu=1,
        n_ctx=4096,
    )

    # Load the tokenizer
    tokenizer = load_tokenizer()

    messages = [
        {
            "role": "system",
            "content": 'You are "BigMak", a superintelligent artificial intelligence and your purpose is to be a software development assitant. Focus on showing code, and keep other text minimal.',
        }
    ]

    while True:
        generated_token_ids = []
        prompt = input("Q:")
        add_to_messages(messages=messages, prompt=prompt)
        tokenized_ids = generate_token_ids(messages=messages)
        for token in llm.generate(
            tokenized_ids, top_k=40, top_p=0.95, temp=0.1, repeat_penalty=1.1
        ):
            if token == 32000:
                break
            generated_token_ids.append(token)
            piece = llm.detokenize([token]).decode()
            print(piece, end="", flush=True)

        # breakpoint()
        prompt = tokenizer.Decode(generated_token_ids)
        add_to_messages(messages=messages, prompt=prompt, role="assistant")

        print("\n")
