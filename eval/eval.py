import json
from rag import query_vector_store


# Function for evaluating the RAG system
def eval(persistent_directory, args):
    # Load the json file with evaluation data
    with open("eval/evaluation.json", "r") as file:
        tests = json.load(file)

    total_recall = 0
    num_cases = len(tests)

    print("=== Evaluation Started ===")
    print(f"Total test cases: {num_cases}\n")

    for i, test in enumerate(tests, start=1):
        # Get the Questions and Ground Truth files
        query = test["question"]
        files = set(test["files"])

        # Query the vector store
        retrieved_files = query_vector_store(
            query, persistent_directory, args.top_k, args.embedding_model
        )
        retrieved_files = set(retrieved_files)

        # Calculate Recall@10
        num_relevant = len(retrieved_files & files)
        recall_at_10 = num_relevant / len(files)

        print(f"Test {i} - Recall@10: {recall_at_10:.3f}")
        print("-" * 40)

        # Print extra info if needed
        if args.verbose:
            print(f"  Query: {query}")
            print(f"  Ground Truth Files: {files}")
            print(f"  Retrieved Top-10 Files: {retrieved_files}")
            print(f"  Relevant Files Found: {retrieved_files & files}")
            print("-" * 40)

        total_recall += recall_at_10

    # Compute overall Recall@10
    average_recall_at_10 = total_recall / num_cases if num_cases else 0
    print("\n=== Evaluation Complete ===")
    print(f"Overall Recall@10: {average_recall_at_10:.3f}")

    # Save the final result to a log file
    with open("eval/eval_results.log", "a") as log_file:
        log_file.write(
            f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}, Recall@10: {average_recall_at_10}\n")

    print("Results saved to 'eval/eval_results.log'\n")
