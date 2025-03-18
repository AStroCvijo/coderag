# Extensions
extensions = [".npmrc", ".nvmdrc", ".md", ".yml", ".CN", '.js', ".json", ".1", ".vue", "LICENSE", ".py", ".sh", ".csv"]

# List of OpenAI embedding models with their max dimensions
OPENAI_MODELS = ['text-embedding-3-large','text-embedding-3-small', 'text-embedding-ada-002',]

# List of OpenAI models
LLMS = ['gpt-4', 'gpt-3.5-turbo', 'gpt-4o-mini']

# OpenAI model context length
MODEL_CONTEXT_LENGTHS = {
    "gpt-3.5-turbo": 16385,
    "gpt-4o-mini": 128000,
}