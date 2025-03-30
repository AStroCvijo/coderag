from langchain_openai import ChatOpenAI
from utils.const import LLMS, OPENAI_MODELS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline

def get_llm(model_name, max_tokens=1200):
    # Check if the model is OpenAI
    if model_name in LLMS:
        return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=max_tokens)
    else:
        try:
            # Load Hugging Face model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

            # Create a text-generation pipeline
            text_generation_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens
            )

            # Wrap the pipeline in a LangChain-compatible format
            return HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        except Exception as e:
            raise ValueError(f"Failed to load LLM model, or model doesn't exist '{model_name}': {e}")

def get_embeddings_function(embedding_model):
    # Check if the model is OpenAI
    if embedding_model in OPENAI_MODELS:
        return OpenAIEmbeddings(model=embedding_model)
    else:
        try:
            return HuggingFaceEmbeddings(model_name=embedding_model)
        except Exception as e:
            raise ValueError(f"Failed to load embedding model, or model doesn't exist '{embedding_model}': {e}")