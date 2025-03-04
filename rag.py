import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def expand_query(query):
    llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=19)
    prompt = f"Expand the following query to improve document retrieval. Provide only the expanded query, without any extra information:\n\n{query}"
    expanded_query = llm.invoke(prompt)
    print(expanded_query.content)
    return expanded_query.content

def generate_summary(code):
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.0,
                     max_tokens=200)
    
    # Adding a more specific instruction to ensure structured output
    prompt = f"""
    Summarize the following code in a structured and concise manner. 
    Provide only the summary and avoid any additional explanations. 
    The summary should focus on the main logic and key functions, if any.

    Code:
    {code}

    Summary:
    """
    
    summary = llm.invoke(prompt)
    
    # Ensuring the output is only the summary content
    return summary.content.strip()

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

    print(f"Found {len(code_files)} files with extensions {extensions}")

    for file_path in code_files:
        try:
            # Remove the 'data/' prefix from the file path for storage
            file_path_relative = file_path.replace(f"{data_path}/", "")

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                summary = generate_summary(content)
                metadata = {"source": file_path_relative, "extension": os.path.splitext(file_path)[1]}
                docs.append(Document(page_content=summary, metadata=metadata))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return docs

# Function for creating the vector store
def create_vector_store(data_path, extensions, persistent_directory, args):
    print("Loading files...")
    docs = load_files_as_documents(data_path, extensions)

    if not docs:
        print("No documents found. Exiting.")
        return

    # Split code into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    split_docs = text_splitter.split_documents(docs)

    print(f"Number of chunks created: {len(split_docs)}")

    if not split_docs:
        print("No chunks created. Exiting.")
        return

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create the vector store
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    
    print(f"Vector store created at {persistent_directory}")  

def query_vector_store(query, persistent_directory, args):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the Chroma database
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Use the 'similarity' search type
    retriever = db.as_retriever(search_type=args.search_type, search_kwargs={"k": 20})

    # Retrieve documents
    docs = retriever.invoke(query)

    # Sort the retrieved documents by their relevance score in descending order
    sorted_docs = sorted(docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True)

    # Select only the top 10 documents after sorting
    top_docs = sorted_docs[:10]

    # Extract the unique file names from the documents
    retrieved_files = list(set([doc.metadata.get("source") for doc in top_docs]))

    return retrieved_files

