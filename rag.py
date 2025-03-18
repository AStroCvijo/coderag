import os
import ast
import torch
from generate import *
from pathlib import Path
from langchain_chroma import Chroma
from langchain.schema import Document
from utils.const import OPENAI_MODELS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------------------------------------------
# EMBEDDING MODEL DETECTION
# ------------------------------------------------------------------------------------------------------

def get_embeddings_function(embedding_model):
    """Returns the appropriate embeddings function based on the model type."""
    if embedding_model in OPENAI_MODELS:
        return OpenAIEmbeddings(model=embedding_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = AutoModel.from_pretrained(embedding_model)
        
        def hf_embeddings(texts):
            tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                output = model(**tokens)
            return output.last_hidden_state.mean(dim=1).tolist()
        
        return hf_embeddings

# ------------------------------------------------------------------------------------------------------
# VECTORSTORE CREATION
# ------------------------------------------------------------------------------------------------------

# Function for generating code summaries
def generate_summary(code):
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=400)
    
    # Initialize prompt template
    prompt = f"""
    Analyze the following code and create a structured summary containing:
    1. Key functionality and purpose (1-2 sentences)
    2. Main classes/functions with brief descriptions
    3. Important dependencies/imports
    4. Notable algorithms/patterns
    5. Key input/output relationships

    Format using bullet points with minimal markdown. Be concise but technically precise.

    Code:
    {code}

    Summary:
    """
    
    # Return code summary
    summary = llm.invoke(prompt)
    return summary.content.strip()

# Function for extracting file metadata
def extract_code_metadata(code):
    try:
        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        return functions, classes
    except Exception:
        return [], []

# Function to recursively get all code files with specified extension in a folder
def get_code_files(data_path, extensions):
    code_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                code_files.append(os.path.join(root, file))
    return code_files

# Function to process a single file
def process_file(file_path, data_path, llm_summary):
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        relative_path = str(Path(file_path).relative_to(data_path))
        absolute_path = str(Path(file_path).resolve())
        file_extension = Path(file_path).suffix.lower()
        functions, classes = extract_code_metadata(content)

        metadata = {
            "source": relative_path,
            "absolute_path": absolute_path,
            "extension": file_extension,
            "classes": ", ".join(classes) if classes else "None",
            "functions": ", ".join(functions) if functions else "None",
        }

        if llm_summary:
            summary = generate_summary(content)
            return Document(
                page_content=f"SUMMARY: {summary}\n\nCODE SNIPPET: {content}...",
                metadata=metadata
            )
        else:
            return Document(
                page_content=content,
                metadata=metadata
            )
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Function to load files and process them in parallel
def load_files_as_documents(data_path, extensions, llm_summary=False):
    code_files = get_code_files(data_path, extensions)
    docs = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file, data_path, llm_summary): file for file in code_files}

        for future in as_completed(future_to_file):
            doc = future.result()
            if doc:  # Ignore failed cases
                docs.append(doc)

    return docs

# Function for creating the vector store
def create_vector_store(data_path, extensions, persistent_directory, chunk_size, chunk_overlap, embedding_model, llm_summary):
    # Load all the docs
    print("Loading files... (this may take a while)")
    docs = load_files_as_documents(data_path, extensions, llm_summary)

    if not docs:
        print("No documents found. Exiting.")
        return

    # Split code into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=chunk_overlap,
                                                   add_start_index=True)
    split_docs = text_splitter.split_documents(docs)

    print(f"Number of chunks created: {len(split_docs)}")

    if not split_docs:
        print("No chunks created. Exiting.")
        return

    # Get embeddings function based on model type
    embeddings = get_embeddings_function(embedding_model)

    # Create the Chroma vector store
    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created at {persistent_directory}")

# ------------------------------------------------------------------------------------------------------
# VECTORSTORE QUERY
# ------------------------------------------------------------------------------------------------------

# Function for querying the vector store
def query_vector_store(query, persistent_directory, k, embedding_model):
    # Get embeddings function based on model type
    embeddings = get_embeddings_function(embedding_model)

    # Load the Chroma database
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Perform similarity search
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Retrieve top-k documents
    docs = retriever.invoke(query)

    # Ensure unique documents before sorting
    unique_docs = {doc.metadata.get("source"): doc for doc in docs}.values()

    # Sort by relevance score in descending order
    sorted_docs = sorted(unique_docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True)

    # Select only the top 40 documents
    top_docs = sorted_docs[:40]

    # Extract file names
    retrieved_files = [doc.metadata.get("source") for doc in top_docs]

    return retrieved_files

# Function for generating LLM summaries of the retrieved code files
def query_vector_store_with_llm(query, persistent_directory, embedding_model):
    # Get embeddings function based on model type
    embeddings = get_embeddings_function(embedding_model)

    # Load the Chroma database
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Use the 'similarity' search type
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Create a new workflow instance
    workflow = StateGraph(GraphState)

    # Add nodes to the workflow
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("out_of_scope_response", out_of_scope_response)

    # Set the entry point and define the edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "out_of_scope": "out_of_scope_response",
            "generate": "generate",
        },
    )
    workflow.add_edge("out_of_scope_response", END)
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    # Compile the workflow
    app = workflow.compile()

    # Genrate the answer
    inputs = {"question": query}
    for output in app.stream(inputs):
        for key, value in output.items():
            final_output = value

    return final_output["generation"]

# ------------------------------------------------------------------------------------------------------
# METHODS NOT USEFUL FOR THIS SPECIFIC TASK
# ------------------------------------------------------------------------------------------------------

# Function for expanding the user query
def expand_query(query):
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=100)

    # Initialize prompt template
    prompt = f"""Optimize the following query to better suit a 
    Retrieval-Augmented Generation (RAG) system, where the vector store is
    constructed from code files that have been summarized by a Large Language Model (LLM). 
    Ensure the query aligns with the summarized nature of the data, improving retrieval accuracy and relevance.

    Return only the enhanced query and no other text.

    Original query: {query}
    """
    
    # Return expanded query
    expanded_query = llm.invoke(prompt)
    return expanded_query.content.strip('"').strip()

# Function for reranking with LLM
def rerank_with_llm(query, documents):
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    ranked_docs = []
    for doc in documents:
        # Initialize prompt template
        prompt = f"""
        Given the query: "{query}"
        Rank the relevance of the following retrieved document on a scale of 1 to 100:
        
        {doc.page_content[:2000]}

        Provide only a numeric score (1-100) with no extra text.
        """
        score = llm.invoke(prompt).content.strip()
        
        try:
            doc.metadata["rerank_score"] = float(score)
            ranked_docs.append(doc)
        except ValueError:
            continue
    
    # Return sorted docs
    return sorted(ranked_docs, key=lambda x: x.metadata["rerank_score"], reverse=True)