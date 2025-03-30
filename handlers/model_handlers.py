import torch
from langchain_openai import ChatOpenAI
from utils.const import LLMS, OPENAI_MODELS
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm(model_name, max_tokens=1200):
    if model_name in LLMS:
        return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=max_tokens)
    else:
        # Implement llm handling as necessary...
        pass

def get_embeddings_function(embedding_model):
    # Check if the model is OpenAI
    if embedding_model in OPENAI_MODELS:
        return OpenAIEmbeddings(model=embedding_model)
    else:
        # Create and return Hugging Face Embeddings with extended settings
        return  HuggingFaceEmbeddings(
            model_name=embedding_model,
        )