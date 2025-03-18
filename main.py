import os
from rag import *
from utils.repo import *
from eval.eval import eval
from utils.const import extensions
from utils.argparser import arg_parse
from ui.main_ui import start_ui

if __name__ == "__main__":
    # Parse the arguments
    args = arg_parse()

    # Start UI if specified
    if args.user_interface:
        start_ui(args)

    # Start evaluation if specified
    if args.eval:
        # Clone GitHub repo
        repo_url = args.repo_url
        data_path = os.path.join("data", extract_repo_name(repo_url))
        clone_repo(repo_url, data_path)

        # Create the vector store
        persistent_directory = os.path.join("db", f"chroma_{extract_repo_name(repo_url)}_{args.chunk_size}_{args.chunk_overlap}_{args.embedding_model}_{args.llm}")
        if not os.path.exists(persistent_directory):
            create_vector_store(data_path, extensions, persistent_directory, args.chunk_size, args.chunk_overlap, args.embedding_model, args.llm_summary, args.llm)
        else:
            print("Vector store with those settings already exists.")

        # Evaluate on evaluation.json
        eval(persistent_directory, args)