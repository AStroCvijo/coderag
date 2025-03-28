import torch
from langchain_openai import ChatOpenAI
from utils.const import LLMS, OPENAI_MODELS
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModel, AutoTokenizer

def get_llm(model_name):
    if model_name in LLMS:
        return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=400)
    else:
        # Implement llm handling as necessary...
        pass

def get_embeddings_function(embedding_model):
    # Check if the model is OpenAI
    if embedding_model in OPENAI_MODELS:
        return OpenAIEmbeddings(model=embedding_model)
    else:
        # Implement embedding model handling as necessary...
        # Load Hugging Face model dynamically
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = AutoModel.from_pretrained(embedding_model)

        # Define the embeddings function for Hugging Face models
        def embeddings(texts):
            tokens = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                output = model(**tokens)
            return output.last_hidden_state.mean(dim=1).tolist()