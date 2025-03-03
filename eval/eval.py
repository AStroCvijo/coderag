import json
from rag import query_vector_store

def eval(persistent_directory, args):
    with open("eval/evaluation.json", 'r') as file:
        tests = json.load(file)

    for test in tests:
        query = test["question"]
        files = test["files"]

        # Query the vector store
        retrieved_files = query_vector_store(query, persistent_directory, args)

        # Calculate the accuracy
        correct_files_count = 0
        wrong_files_count = 0
        for file in retrieved_files:
            if file in files:
                correct_files_count += 1
            else:
                wrong_files_count += 1

        if args.verbose:
            # Print the question, correct answer, your answer (retrieved_files), and the file counts
            print("=" * 80)
            print(f"Question:\n{query}")
            print("-" * 80)
            print(f"Correct Answer:\n{', '.join(files)}")
            print(f"Your Answer (Retrieved Files):\n{', '.join(retrieved_files)}")
            print("-" * 80)
                
            # Summary of file counts
            print(f"Number of Correct Files: {correct_files_count}")
            print(f"Number of Wrong Files: {wrong_files_count}")
            print("=" * 80)
            print("\n")
        else:
            # Summary of file counts            
            print("=" * 80)
            print(f"Number of Correct Files: {correct_files_count}")
            print(f"Number of Wrong Files: {wrong_files_count}")
            print("=" * 80)
            print("\n")