import json
from rag import query_vector_store
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Function for evaluating the RAG system
def eval(persistent_directory, args):
    # Load the json file with evaluation data
    with open("eval/evaluation.json", "r") as file:
        tests = json.load(file)

    total_recall = 0
    num_cases = len(tests)

    # Print evaluation header
    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== RAG Evaluation Started ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Configuration:{Style.RESET_ALL}")
    print(f"  - Embedding Model: {args.embedding_model}")
    print(f"  - Top K: {args.top_k}")
    print(f"  - Chunk Size: {args.chunk_size}")
    print(f"  - Chunk Overlap: {args.chunk_overlap}")
    print(f"\n{Fore.YELLOW}Test Cases:{Style.RESET_ALL} {num_cases}\n")

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
        
        # Color code the recall score
        recall_color = Fore.GREEN if recall_at_10 >= 0.7 else Fore.YELLOW if recall_at_10 >= 0.4 else Fore.RED
        recall_display = f"{recall_color}{recall_at_10:.3f}{Style.RESET_ALL}"

        print(f"{Fore.BLUE}{Style.BRIGHT}Test Case {i}:{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}Recall@10:{Style.RESET_ALL} {recall_display}")
        print(f"  {Fore.WHITE}Relevant Files Found:{Style.RESET_ALL} {num_relevant}/{len(files)}")

        # Print extra info if needed
        if args.verbose:
            print(f"\n  {Fore.MAGENTA}Query:{Style.RESET_ALL} {query}")
            print(f"  {Fore.MAGENTA}Ground Truth Files:{Style.RESET_ALL} {files}")
            print(f"  {Fore.MAGENTA}Retrieved Top-{args.top_k} Files:{Style.RESET_ALL} {retrieved_files}")
            print(f"  {Fore.MAGENTA}Matching Files:{Style.RESET_ALL} {retrieved_files & files}")
        
        print(f"{Fore.LIGHTBLACK_EX}{'-'*60}{Style.RESET_ALL}")
        total_recall += recall_at_10

    # Compute overall Recall@10
    average_recall_at_10 = total_recall / num_cases if num_cases else 0
    
    # Color code the final score
    final_color = Fore.GREEN if average_recall_at_10 >= 0.7 else Fore.YELLOW if average_recall_at_10 >= 0.4 else Fore.RED
    final_display = f"{final_color}{average_recall_at_10:.3f}{Style.RESET_ALL}"

    print(f"\n{Fore.CYAN}{Style.BRIGHT}=== Evaluation Results ==={Style.RESET_ALL}")
    print(f"{Fore.WHITE}Overall Recall@10:{Style.RESET_ALL} {final_display}")
    print(f"{Fore.WHITE}Total Test Cases:{Style.RESET_ALL} {num_cases}")

    # Save the final result to a log file
    with open("eval/eval_results.log", "a") as log_file:
        log_file.write(
            f"Config: chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, "
            f"model={args.embedding_model}, top_k={args.top_k} | "
            f"Recall@10: {average_recall_at_10:.3f}\n"
        )

    print(f"\n{Fore.GREEN}Results saved to {Style.BRIGHT}'eval/eval_results.log'{Style.RESET_ALL}\n")