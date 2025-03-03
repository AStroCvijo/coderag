import os
import getpass

from rag import *
from utils.argparser import arg_parse
from utils.const import extensions
from eval.eval import eval

# Get the OpenAI API
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if __name__ == "__main__":

    # Parse the arguments
    args = arg_parse()

    # Clone the GitHub repo
    repo_url = args.repo_url
    data_path = "data"
    clone_repo(repo_url, data_path)

    # Persistent directory path
    persistent_directory = os.path.join("db", "chroma_db_code")

    # Create the vector store if it doesn't exist already
    if not os.path.exists(persistent_directory):
        create_vector_store(data_path, extensions, persistent_directory)
    else:
        print("Vector store already exists.")

    # Evaluate on evaluation.json
    if args.eval:
        eval(persistent_directory, args)

    # Allow the user to query over the databse
    if args.query:
        # Implement UI for user query
        print("Query......")
