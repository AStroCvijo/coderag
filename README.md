
# **LLM Listwise Reranker for CodeRAG**

Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task

Developed as part of an application for a JetBrains internship.

## ⚡**Quickstart**

### **Prerequisites**
- Python 3.8+
- Conda (for environment management)

### 🛠️**Setup**
Clone the repository and install dependencies:

```bash
git clone https://github.com/AStroCvijo/llm_listwise_reranker_for_coderag.git
cd llm_listwise_reranker_for_coderag
conda create -n rag python=3.8
conda activate rag
pip install -r requirements.txt
```

### 🖥️**Run with User Interface**
```bash
python main.py -ui
```

### 📊**Evaluation**
To evaluate the model on the provided test set:

```bash
chmod +x eval.sh
./eval.sh
```

## 📖**Arguments Guide**

| Argument | Description | Default | Options |
|----------|------------|---------|---------|
| `-k`, `--top_k` | Number of results retrieved per query | `30` | Any positive integer |
| `-cs`, `--chunk_size` | Text chunk size for indexing | `1200` | Any positive integer |
| `-co`, `--chunk_overlap` | Overlapping tokens between chunks | `200` | Any non-negative integer |
| `-ls`, `--llm_summary` | Enable document summarization | `True` | `True`, `False` |
| `-em`, `--embedding_model` | Embedding model selection | `text-embedding-3-large` | `text-embedding-3-large`, `text-embedding-3-small`, `text-embedding-ada-002` |
| `-m`, `--llm` | LLM model to use | `gpt-4o-mini` | `gpt-3.5-turbo`, `gpt-4o-mini` |
| `-ui`, `--user_interface` | Enable UI mode | `False` | `True`, `False` |
| `-v`, `--verbose` | Enable debugging output | `False` | `True`, `False` |
| `-e`, `--eval` | Enable evaluation mode | `False` | `True`, `False` |
| `-ru`, `--repo_url` | Repository URL for indexing | `https://github.com/viarotel-org/escrcpy` | Any valid repo URL |

## 📄**Experiments**

For detailed experiments and evaluations of this RAG system, please refer to the following document:

[Experiments and Evaluations](https://astrocvijo.github.io/llm_listwise_reranker_for_coderag/docs/experiments.pdf)

This document provides comprehensive insights into the performance, benchmarks, and various configurations tested during the development of the LLM Listwise Reranker for CodeRAG. It includes comparisons of different embedding models, chunk sizes, and other parameters to help you understand the system's capabilities and optimize it for your specific use case.

## 🧩**Seamless Integration with Multiple LLM and Embedding Providers (Local/API/Custom)**

Currently, `utils/handlers.py` contains the handlers for LLM and embedding models, but only OpenAI models are implemented. To use a different LLM or embedding model, you can modify the logic in `utils/handlers.py` accordingly.

**Note:** Models that support structured output and have a context length of at least **16,385** tokens are preferred.
