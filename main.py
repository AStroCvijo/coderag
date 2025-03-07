import os
import getpass

from rag import *

from utils.argparser import arg_parse
from utils.const import extensions
from utils.repo import *

from eval.eval import eval

from user_interface.main_ui import start_ui

# Get the OpenAI API
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if __name__ == "__main__":

    # Parse the arguments
    args = arg_parse()

    # Clone the GitHub repo
    repo_url = args.repo_url
    data_path = os.path.join("data", extract_repo_name(repo_url))
    clone_repo(repo_url, data_path)

    # Persistent directory path
    persistent_directory = os.path.join("db", f"chroma_{extract_repo_name(repo_url)}_{args.chunk_size}_{args.chunk_overlap}")

    # Create the vector store if it doesn't exist already
    if not os.path.exists(persistent_directory):
        create_vector_store(data_path, extensions, persistent_directory, args)
    else:
        print("Vector store already exists.")

    # Evaluate on evaluation.json
    if args.eval:
        eval(persistent_directory, args)

    # Start UI
    # if args.query:
