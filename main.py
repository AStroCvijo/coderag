from git import Repo
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Function to recursively get all code files with specified extension in a folder
def get_code_files(data_path, extensions):
    code_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if extensions is None or file.endswith(tuple(extensions)):
                code_files.append(os.path.join(root, file))
    return code_files

# Function to load files, extract their contents and metadata
def load_files_as_documents(data_path, extensions):
    docs = []
    code_files = get_code_files(data_path, extensions)

    print(f"Found {len(code_files)} files with extensions {extensions}")

    for file_path in code_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                metadata = {"source": file_path, "extension": os.path.splitext(file_path)[1]}
                docs.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return docs

# Function for creating the vector store
def create_vector_store(data_path, extensions, persistent_directory):
    print("Loading files...")
    docs = load_files_as_documents(data_path, extensions)

    if not docs:
        print("No documents found. Exiting.")
        return

    # Split code into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    print(f"Number of chunks created: {len(split_docs)}")

    if not split_docs:
        print("No chunks created. Exiting.")
        return

    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Debug: Check if embeddings are being generated
    sample_embedding = embeddings.embed_query("This is a test sentence.")
    if not sample_embedding:
        print("Failed to generate embeddings. Exiting.")
        return

    # **FIX: Use 'persist_directory' instead of 'persistent_directory'**
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
    
    print(f"Vector store created at {persistent_directory}")  

def query_vector_store(query):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Use the 'similarity_score_threshold' retriever to get scores
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.08, "k": 10})
    
    # Retrieve documents with scores
    relevant_docs_with_scores = retriever.invoke(query)

    # Sort documents by score in descending order
    sorted_docs = sorted(relevant_docs_with_scores, key=lambda x: x.metadata.get("score", 0.0), reverse=True)


    # Use a set to store unique file names
    retrieved_files = set([doc.metadata.get("source") for doc in sorted_docs])

    # Convert set to list before returning
    return list(retrieved_files)

if __name__ == "__main__":
    # Repo url
    repo_url = "https://github.com/viarotel-org/escrcpy"

    # Data path
    data_path = "data"

    # Clone the github repo
    if not os.path.exists(data_path):
        Repo.clone_from(repo_url, data_path)
        print(f"Repository cloned to {data_path}")
    else:
        print("Repo already cloned!!!")

    # Extensions
    extensions = [".py", ".vue", ".js", ".json", ".md"]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(current_dir, "db")
    persistent_directory = os.path.join(db_dir, "chroma_db_code")

    if not os.path.exists(persistent_directory):
        create_vector_store(data_path=data_path, extensions=extensions, persistent_directory=persistent_directory)
    else:
        print("Vector store already exists!!!")

    query = "How does the SelectDisplay component handle the device options when retrieving display IDs?"
    retrieved_files = query_vector_store(query=query)
    
    print()
    print(query)
    print()
    for file in retrieved_files:
        print(file)
