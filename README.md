
# LLM Listwise Reranker for CodeRAG

Project made for the purpose of applying to JetBrains internship.  

Retrieval-Augmented Generation (RAG) system over a code repository for a question-answering task.

Date of creation: March, 2024

## Quickstart
1. Setup
    ```bash
    git clone https://github.com/AStroCvijo/llm_listwise_reranker_for_coderag.git
	cd llm_listwise_reranker_for_coderag
	conda create -n rag
	pip install -r requirements.txt
	```
2. Run with user interface
    ```bash
    python main.py -ui
    ```
## Evaluation
To evaluate the performance on the provided test with default settings run the following commands
```bash
chmod +x eval.sh
./eval.sh
```

## Arguments Guide

### RAG (Retrieval-Augmented Generation) Arguments

`-k or --top_k`  
Specify the number of results to retrieve from the index for each query. Default: `30`.  
Range: Any positive integer.

`-cs or --chunk_size`  
Define the size of each text chunk for indexing. Default: `1200`.  
Range: Any positive integer.

`-co or --chunk_overlap`  
Specify the number of overlapping tokens between consecutive chunks. Default: `200`.  
Range: Any non-negative integer.

`-ls or --llm_summary`  
Enable or disable summarization of documents before passing them to the model. Default: `True`.  
Options: `True` or `False`.

`-em or --embedding_model`  
Specify the embedding model to use. Default: `text-embedding-3-large`.  
Options:
- `text-embedding-3-large`
- `text-embedding-3-small`
- `text-embedding-ada-002`
- `text-embedding-babbage-001`
- `text-embedding-curie-001`
- `text-embedding-davinci-002`

### Other Arguments

`-ui or --user_interface`  
Enable the user interface. Default: `False`.  
Options: `True` (for enabling UI) or `False` (for disabling UI).

`-v or --verbose`  
Enable verbose mode for debugging output. Default: `False`.  
Options: `True` or `False`.

`-e or --eval`  
Enable model evaluation mode. Default: `False`.  
Options: `True` or `False`.

`-m or --model`  
Specify the model to use. Default: `gpt-3.5-turbo`.  
Options:
- `gpt-4`
- `gpt-3.5-turbo`
- `gpt-4o-mini`

`-ru or --repo_url`  
Specify the repository URL to use. Default: `https://github.com/viarotel-org/escrcpy`.  
Example: `https://github.com/viarotel-org/escrcpy`

# Experiments

For detailed experiments and evaluations of this RAG system, please refer to the following document:

[Experiments and Evaluations](https://astrocvijo.github.io/llm_listwise_reranker_for_coderag/docs/experiments.pdf)

This document provides comprehensive insights into the performance, benchmarks, and various configurations tested during the development of the LLM Listwise Reranker for CodeRAG. It includes comparisons of different embedding models, chunk sizes, and other parameters to help you understand the system's capabilities and optimize it for your specific use case.
