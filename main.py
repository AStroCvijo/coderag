import os
import sys
import getpass

from rag import *
from utils.repo import *
from eval.eval import eval
from utils.const import extensions
from utils.argparser import arg_parse
from user_interface.main_ui import start_ui

# Get the OpenAI API
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if __name__ == "__main__":

    # Parse the arguments
    args = arg_parse()

    # Start UI if specified
    if args.user_interface:
        start_ui(args)
        sys.exit(0)

    # Clone GitHub repo
    repo_url = args.repo_url
    data_path = os.path.join("data", extract_repo_name(repo_url))
    clone_repo(repo_url, data_path)

    # Create the vector store
    persistent_directory = os.path.join("db", f"chroma_{extract_repo_name(repo_url)}_{args.chunk_size}_{args.chunk_overlap}")
    if not os.path.exists(persistent_directory):
        create_vector_store(data_path, extensions, persistent_directory, args.chunk_size, args.chunk_overlap, args.embedding_model, args.llm_summary)
    else:
        print("Vector store with those settings already exists.")

    # Evaluate on evaluation.json
    if args.eval:
        eval(persistent_directory, args)
