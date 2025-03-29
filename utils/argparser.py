import argparse

# Function for parsing arguments
def arg_parse():
    parser = argparse.ArgumentParser()

    # RAG arguments
    parser.add_argument("-k", "--top_k", type=int, default=40)
    parser.add_argument("-cs", "--chunk_size", type=int, default=1200)
    parser.add_argument("-co", "--chunk_overlap", type=int, default=200)
    parser.add_argument("-ls", "--llm_summary", type=bool, default=True)
    parser.add_argument(
        "-em",
        "--embedding_model",
        type=str,
        choices=[
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        default="text-embedding-3-large",
    )

    # User interface argumenrts
    parser.add_argument(
        "-ui",
        "--user_interface",
        action="store_true",
        default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)

    parser.add_argument("-e", "--eval", action="store_true", default=False)
    parser.add_argument(
        "-m",
        "--llm",
        type=str,
        choices=["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini", "EleutherAI/pythia-160m",],
        default="gpt-4o-mini",
    )
    parser.add_argument(
        "-ru",
        "--repo_url",
        type=str,
        default="https://github.com/viarotel-org/escrcpy")

    # Parse the arguments
    return parser.parse_args()
