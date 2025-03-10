import json
from rag import query_vector_store

def eval(persistent_directory, args):
    with open("eval/evaluation.json", 'r') as file:
        tests = json.load(file)

    total_recall = 0
    num_cases = len(tests)

    for i, test in enumerate(tests, start=1):
        query = test["question"]
        files = set(test["files"])  # Convert ground truth files to set

        # Query the vector store
        retrieved_files = query_vector_store(query, persistent_directory)
        retrieved_files = set(retrieved_files)

        # Calculate Recall@10
        num_relevant = len(retrieved_files & files)
        recall_at_10 = num_relevant / len(files)

        print(f"Test {i}")
        print(f"  Recall@10: {recall_at_10:.3f}\n")

        if args.verbose:
            print(f"  Query: {query}")
            print(f"  Ground Truth Files: {files}")
            print(f"  Retrieved Top-10 Files: {retrieved_files}")
            print(f"  Relevant Files Found: {retrieved_files & files}")

        total_recall += recall_at_10
    
    # Compute overall Recall@10
    average_recall_at_10 = total_recall / num_cases if num_cases else 0
    print(f"Recall@10: {average_recall_at_10:.3f}")

    # Save the final result to a log file
    with open("eval/eval_results.log", "a") as log_file:
        log_file.write(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}, Recall@10: {average_recall_at_10}\n")
