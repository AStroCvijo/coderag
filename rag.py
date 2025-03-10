import os
import ast
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

extension_to_language = {
    ".py": "Python",
    ".js": "JavaScript",
    ".cpp": "C++",
    ".java": "Java",
    ".go": "Go",
    ".rb": "Ruby",
    ".php": "PHP",
    ".html": "HTML",
    ".css": "CSS",
    ".ts": "TypeScript",
    ".swift": "Swift",
    ".json": "JSON",
    ".md": "Markdown",
    ".yml": "YAML",
    ".vue": "VUE",
}

def get_language(file_extension):
    # Return the language from the dictionary or 'Unknown' if the extension is not found
    language = extension_to_language.get(file_extension, "Unknown")
    if language == "Unknown":
        print(f"Warning: Unknown file extension '{file_extension}' encountered.")
    return language

def expand_query(query):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=100)
    prompt = f"""Optimize the following query to better suit a 
    Retrieval-Augmented Generation (RAG) system, where the vector store is
    constructed from code files that have been summarized by a Large Language Model (LLM). 
    Ensure the query aligns with the summarized nature of the data, improving retrieval accuracy and relevance.

    Return only the enhanced query and no other text.

    Original query: {query}
    """
    
    expanded_query = llm.invoke(prompt)
    return expanded_query.content.strip('"').strip()

def generate_summary(code):
    """Generate structured code summary with language-specific context"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=400)
    
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

# Function to recursively get all code files with specified extension in a folder
def get_code_files(data_path, extensions):
    code_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                code_files.append(os.path.join(root, file))
    return code_files

# Function to load files and extract their contents and metadata
def load_files_as_documents(data_path, extensions):
    docs = []
    code_files = get_code_files(data_path, extensions)
    
    for file_path in code_files:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
                
        relative_path = str(Path(file_path).relative_to(data_path))
        file_extension = Path(file_path).suffix.lower()
        language = get_language(file_extension)
        functions, classes = extract_code_metadata(content)
        summary = generate_summary(content)

        metadata = {
            "source": relative_path,
            "extension": file_extension,
            "language": language,
            "functions": ", ".join(functions) if functions else "None",
            "classes": ", ".join(classes) if classes else "None",
        }
            
        docs.append(Document(
            page_content=f"SUMMARY: {summary}\n\nCODE SNIPPET: {content[:2000]}...",
            metadata=metadata
        ))
            
    return docs

# Function for creating the vector store
def create_vector_store(data_path, extensions, persistent_directory, chunk_size, chunk_overlap):
    print("Loading files... (this may take a while)")
    docs = load_files_as_documents(data_path, extensions)

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

    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072,
        show_progress_bar=True
    )

    # Create the Chroma vector store
    db = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persistent_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created at {persistent_directory}")  

def query_vector_store(query, persistent_directory):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=3072)
    
    # Load the Chroma database
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Use the 'similarity' search type
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 30})

    # Retrieve documents
    docs = retriever.invoke(query)

    # Sort the retrieved documents by their relevance score in descending order
    sorted_docs = sorted(docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True)

    # Select only the top 10 documents after sorting
    top_docs = sorted_docs[:10]

    # Extract the unique file names from the documents
    retrieved_files = list(set([doc.metadata.get("source") for doc in top_docs]))

    return retrieved_files