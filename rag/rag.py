import os
import ast
from tqdm import tqdm
from rag.generate import *
from pathlib import Path
from langchain_chroma import Chroma
from langchain.schema import Document
from colorama import init, Fore, Style
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from handlers.model_handlers import get_llm, get_embeddings_function

# Initialize colorama
init(autoreset=True)

os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# ------------------------------------------------------------------------------------------------------
# VECTORSTORE CREATION
# ------------------------------------------------------------------------------------------------------

# Function for generating code summaries
def generate_summary(code, relative_path, model_name):
    # Initialize LLM
    llm = get_llm(model_name)

    # Initialize prompt template
    prompt = f"""
    Analyze the following code and file path and create a structured summary containing:
    1. Key functionality and purpose (1-2 sentences)
    2. Main classes/functions with brief descriptions
    3. Important dependencies/imports
    4. Notable algorithms/patterns
    5. Key input/output relationships
    6. The general purpose of the file

    Format using bullet points with minimal markdown. Be concise but technically precise.

    Code:
    {code}

    Path:
    {relative_path}
    Summary:
    """

    # Return code summary
    summary = llm.invoke(prompt)
    return summary.content.strip()


# Function for extracting file metadata
def extract_code_metadata(code):
    try:
        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(
            tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(
            tree) if isinstance(node, ast.ClassDef)]
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
def process_file(file_path, data_path, llm_summary, llm):
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        relative_path = str(Path(file_path).relative_to(data_path))
        absolute_path = str(Path(file_path).resolve())
        file_extension = Path(file_path).suffix.lower()
        functions, classes = extract_code_metadata(content)
        # purpose = generate_purpose(llm, relative_path, file_extension, functions, classes)

        metadata = {
            "source": relative_path,
            "absolute_path": absolute_path,
            "extension": file_extension,
            "classes": ", ".join(classes) if classes else "None",
            "functions": ", ".join(functions) if functions else "None",
            # "purpose": purpose
        }

        if llm_summary:
            summary = generate_summary(content, relative_path, llm)
            return Document(
                page_content=f"SUMMARY: {summary}\n\nCODE SNIPPET: {content}...",
                metadata=metadata,
            )
        else:
            return Document(page_content=content, metadata=metadata)
    except Exception as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Processing {Path(file_path).name}: {str(e)}")
        return None


# Function to load files and process them in parallel
def load_files_as_documents(data_path, extensions, llm, llm_summary=False):
    code_files = get_code_files(data_path, extensions)
    docs = []

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, file, data_path, llm_summary, llm): file
            for file in code_files
        }

        for future in tqdm(
                as_completed(future_to_file),
                total=len(code_files),
                desc=f"{Fore.BLUE}Processing files{Style.RESET_ALL}",
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):
            doc = future.result()
            if doc:
                docs.append(doc)

    return docs


# Function for creating the vector store
def create_vector_store(
    data_path,
    extensions,
    persistent_directory,
    chunk_size,
    chunk_overlap,
    embedding_model,
    llm_summary,
    llm,
):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== Creating Vector Store ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Loading files...{Style.RESET_ALL} (this may take a while)")
    docs = load_files_as_documents(data_path, extensions, llm, llm_summary)

    if not docs:
        print(f"{Fore.RED}No documents found. Exiting.{Style.RESET_ALL}")
        return

    # Split code into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True)
    split_docs = text_splitter.split_documents(docs)

    print(f"{Fore.GREEN}Number of chunks created:{Style.RESET_ALL} {len(split_docs)}")

    if not split_docs:
        print(f"{Fore.RED}No chunks created. Exiting.{Style.RESET_ALL}")
        return

    # Get embeddings function based on model type
    embeddings = get_embeddings_function(embedding_model)

    # Create the Chroma vector store
    print(f"{Fore.YELLOW}Creating vector store...{Style.RESET_ALL}")
    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"{Fore.GREEN}Vector store successfully created at:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{persistent_directory}{Style.RESET_ALL}")

# ------------------------------------------------------------------------------------------------------
# METHODS NOT USEFUL FOR THIS SPECIFIC TASK
# ------------------------------------------------------------------------------------------------------


# Function for expanding the user query
def expand_query(query, model_name):
    # Initialize LLM
    llm = get_llm(model_name)

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
def rerank_with_llm(query, documents, model_name):
    # Initialize LLM
    llm = get_llm(model_name)

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
    return sorted(
        ranked_docs,
        key=lambda x: x.metadata["rerank_score"],
        reverse=True)
