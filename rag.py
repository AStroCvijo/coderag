import os
import ast
import torch
from generate import *
from pathlib import Path
from langchain_chroma import Chroma
from langchain.schema import Document
from utils.const import OPENAI_MODELS
from langchain_openai import ChatOpenAI
from transformers import pipeline, AutoTokenizer
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

# ------------------------------------------------------------------------------------------------------
# VECTORSTORE CREATION
# ------------------------------------------------------------------------------------------------------

def get_embeddings(embedding_model):
    """ Unified embedding handler for both OpenAI and Hugging Face models """
    if embedding_model in OPENAI_MODELS:
        return OpenAIEmbeddings(model=embedding_model)
    else:
        return HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def get_llm(model_name, provider="openai", max_tokens=400):
    """ Unified LLM factory for both providers """
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=max_tokens)
    elif provider == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1
        )
        return HuggingFacePipeline(pipeline=pipe)

def generate_summary(code, llm_model, llm_provider):
    """ Generic summary generation with model switching """
    llm = get_llm(llm_model, llm_provider)
    
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
    
    summary = llm.invoke(prompt)
    return summary.content.strip()

def extract_code_metadata(code):
    try:
        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        return functions, classes
    except Exception:
        return [], []

def get_code_files(data_path, extensions):
    code_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                code_files.append(os.path.join(root, file))
    return code_files

def process_file(file_path, data_path, llm_summary, llm_model, llm_provider):
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
            summary = generate_summary(content, llm_model, llm_provider)
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

def load_files_as_documents(data_path, extensions, llm_summary=False, llm_model="gpt-4", llm_provider="openai"):
    code_files = get_code_files(data_path, extensions)
    docs = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for file in code_files:
            futures.append(
                executor.submit(
                    process_file,
                    file,
                    data_path,
                    llm_summary,
                    llm_model,
                    llm_provider
                )
            )

        for future in as_completed(futures):
            doc = future.result()
            if doc:
                docs.append(doc)

    return docs

def create_vector_store(data_path, extensions, persistent_directory, chunk_size=1000, 
                       chunk_overlap=200, embedding_model="text-embedding-3-small", 
                       llm_summary=False, llm_model="gpt-4", llm_provider="openai"):
    
    print("Loading files...")
    docs = load_files_as_documents(
        data_path, extensions, llm_summary, llm_model, llm_provider
    )

    if not docs:
        print("No documents found")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    split_docs = text_splitter.split_documents(docs)

    print(f"Creating vector store with {len(split_docs)} chunks")
    
    embeddings = get_embeddings(embedding_model)
    
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

def query_vector_store(query, persistent_directory, embedding_model, k=40):
    embeddings = get_embeddings(embedding_model)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k*2}  # Retrieve extra for deduplication
    )

    docs = retriever.invoke(query)
    
    # Deduplicate and sort
    unique_docs = {doc.metadata["source"]: doc for doc in docs}.values()
    sorted_docs = sorted(unique_docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
    
    return [doc.metadata["source"] for doc in sorted_docs[:k]]

def query_vector_store_with_llm(query, persistent_directory, embedding_model, 
                              llm_model="gpt-4", llm_provider="openai"):
    
    embeddings = get_embeddings(embedding_model)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    
    llm = get_llm(llm_model, llm_provider, max_tokens=1000)
    
    # Initialize workflow components with current models
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", lambda state: generate(state, llm))
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("out_of_scope_response", out_of_scope_response)

    # Connect nodes
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"out_of_scope": "out_of_scope_response", "generate": "generate"},
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

    app = workflow.compile()
    
    inputs = {"question": query}
    for output in app.stream(inputs):
        for key, value in output.items():
            final_output = value

    return final_output["generation"]

# ------------------------------------------------------------------------------------------------------
# ADDITIONAL UTILITIES
# ------------------------------------------------------------------------------------------------------

def expand_query(query, llm_model="gpt-4", llm_provider="openai"):
    llm = get_llm(llm_model, llm_provider)
    
    prompt = f"""Optimize this query for RAG retrieval of code summaries:
    Original: {query}
    Enhanced: """
    
    return llm.invoke(prompt).content.strip('"').strip()

def rerank_with_llm(query, documents, llm_model="gpt-4", llm_provider="openai"):
    llm = get_llm(llm_model, llm_provider)
    
    ranked = []
    for doc in documents:
        prompt = f"""
        Query: {query}
        Document: {doc.page_content[:2000]}
        Relevance score (1-100, no text): """
        
        score = llm.invoke(prompt).content.strip()
        try:
            doc.metadata["rerank_score"] = float(score)
            ranked.append(doc)
        except ValueError:
            continue
            
    return sorted(ranked, key=lambda x: x.metadata["rerank_score"], reverse=True)