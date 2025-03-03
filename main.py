import os
import json
import getpass
from dotenv import load_dotenv
from git import Repo

from rag import *

# Get the OpenAI API
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if __name__ == "__main__":

    # Clone the GitHub repo
    repo_url = "https://github.com/viarotel-org/escrcpy"
    data_path = "data"
    clone_repo(repo_url, data_path)

    # Extensions that will be indexed
    extensions = [".py", ".vue", ".js", ".json", ".md"]

    # Persistent directory path
    persistent_directory = os.path.join("db", "chroma_db_code")

    # Create the vector store if it doesn't exist already
    if not os.path.exists(persistent_directory):
        create_vector_store(data_path, extensions, persistent_directory)
    else:
        print("Vector store already exists.")

    with open("evaluation/evaluation.json", 'r') as file:
        tests = json.load(file)

for test in tests:
    query = test["question"]
    files = test["files"]

    # Query the vector store
    retrieved_files = query_vector_store(query, persistent_directory)

    # Calculate the accuracy
    correct_files_count = 0
    wrong_files_count = 0
    for file in retrieved_files:
        if file in files:
            correct_files_count += 1
        else:
            wrong_files_count += 1

    # Print the question, correct answer, your answer (retrieved_files), and the file counts
    print("=" * 80)  # Top separator line for clarity
    print(f"Question:\n{query}")
    print("-" * 80)  # Separator line
    print(f"Correct Answer:\n{', '.join(files)}")
    print(f"Your Answer (Retrieved Files):\n{', '.join(retrieved_files)}")
    print("-" * 80)  # Separator line
    
    # Summary of file counts
    print(f"Number of Correct Files: {correct_files_count}")
    print(f"Number of Wrong Files: {wrong_files_count}")
    print("=" * 80)  # Bottom separator line for clarity
    print("\n")  # Newline between questions

