import os
import validators
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from collections import defaultdict
from utils.repo import *
from utils.const import *
from rag.rag import create_vector_store
from ui.query_ui import start_query_ui

console = Console()

# Function for displaying the banner
def print_banner():
    console.print(
        Panel(
            "[bold cyan]Welcome to the AI-Powered RAG System![/bold cyan] ([italic yellow]Type HELP for a list of commands[/italic yellow])",
            border_style="cyan",
        )
    )

# Function for clearing the screen
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")
    print_banner()

# Function for handling cancel process
def get_input_with_cancel(prompt_text):
    user_input = Prompt.ask(prompt_text)
    if user_input.lower() == "cancel":
        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
        return None
    return user_input

# Function to start the UI
def start_ui(args):
    clear_screen()

    while True:
        # Get user input
        user_input = Prompt.ask("\n[bold blue]Enter command[/bold blue]")

        # Exit/Quit command
        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold red]Goodbye![/bold red] 👋")
            break

        # Index command
        elif user_input.lower() == "index":
            try:
                # Keep prompting until a valid GitHub URL is entered
                while True:
                    repo_url = get_input_with_cancel(
                        "[bold blue]Enter GitHub repository URL[/bold blue]"
                    )
                    if repo_url is None:
                        console.print(
                            "[bold yellow]Indexing canceled by user.[/bold yellow]"
                        )
                        return
                    if validators.url(repo_url) and "github.com" in repo_url:
                        break
                    console.print(
                        "[bold red]Error: The provided URL is not a valid GitHub repository link.[/bold red]"
                    )

                data_path = os.path.join("data", extract_repo_name(repo_url))
                clone_repo(repo_url, data_path)

                # Keep prompting until a valid chunk size is entered
                while True:
                    chunk_size = get_input_with_cancel(
                        "[bold blue]Enter chunk size (positive integer)[/bold blue]"
                    )
                    if chunk_size is None:
                        console.print(
                            "[bold yellow]Indexing canceled by user.[/bold yellow]"
                        )
                        return
                    if chunk_size.isdigit() and int(chunk_size) > 0:
                        chunk_size = int(chunk_size)
                        break
                    console.print(
                        "[bold red]Error: Chunk size must be a positive integer.[/bold red]"
                    )

                # Keep prompting until a valid chunk overlap is entered
                while True:
                    chunk_overlap = get_input_with_cancel(
                        "[bold blue]Enter chunk overlap (non-negative integer)[/bold blue]"
                    )
                    if chunk_overlap is None:
                        console.print(
                            "[bold yellow]Indexing canceled by user.[/bold yellow]"
                        )
                        return
                    if chunk_overlap.isdigit() and int(chunk_overlap) >= 0:
                        chunk_overlap = int(chunk_overlap)
                        break
                    console.print(
                        "[bold red]Error: Chunk overlap must be a non-negative integer.[/bold red]"
                    )

                # Keep prompting until a valid embedding model is entered
                while True:
                    available_models = OPENAI_MODELS + ["Custom Hugging Face Model"]
                    console.print("[bold blue]Available embedding models:[/bold blue]")
                    for i, model in enumerate(available_models, 1):
                        console.print(f"{i}. {model}")
                    
                    model_choice = get_input_with_cancel("[bold blue]Select an embedding model by number (or enter a Hugging Face model name):[/bold blue] ")
                    
                    if model_choice is None:
                        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
                        return
                    
                    if model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models):
                        selected_model = available_models[int(model_choice) - 1]
                        if selected_model == "Custom Hugging Face Model":
                            huggingface_model = get_input_with_cancel("[bold blue]Enter the Hugging Face model name:[/bold blue] ")
                            if huggingface_model:
                                embedding_model = huggingface_model
                                break
                        else:
                            embedding_model = selected_model
                            break
                    
                    elif model_choice:  # Assume user entered a Hugging Face model name directly
                        embedding_model = model_choice.strip()
                        break
                    
                    console.print("[bold red]Error: Please enter a valid number or a Hugging Face model name.[/bold red]")

                # Keep prompting until a valid LLM summary option is entered
                while True:
                    llm_summary = False
                    user_input = get_input_with_cancel(
                        "[bold blue]Use LLM summary? (Y/n):[/bold blue]"
                    )
                    if user_input is None:
                        console.print(
                            "[bold yellow]Indexing canceled by user.[/bold yellow]"
                        )
                        return
                    if user_input.lower() in ["y", "yes"]:
                        llm_summary = True
                        break
                    if user_input.lower() in ["n", "no"]:
                        llm = "NoSummary"
                        break
                    console.print(
                        "[bold red]Error: Please enter 'Y' or 'N' for LLM summary.[/bold red]"
                    )

                if llm_summary:
                    # Keep prompting until a valid embedding model is entered
                    while True:
                        available_llms = LLMS
                        console.print(
                            "[bold blue]Available large language models:[/bold blue]"
                        )
                        for i, llm in enumerate(available_llms, 1):
                            console.print(f"{i}. {llm}")
                        llm_choice = get_input_with_cancel(
                            "[bold blue]Select a LLM by number[/bold blue]"
                        )
                        if llm_choice is None:
                            console.print(
                                "[bold yellow]Indexing canceled by user.[/bold yellow]"
                            )
                            return
                        if llm_choice.isdigit() and 1 <= int(llm_choice) <= len(
                            available_llms
                        ):
                            llm = available_llms[int(llm_choice) - 1]
                            break
                        console.print(
                            "[bold red]Error: Please enter a valid number corresponding to a LLM model.[/bold red]"
                        )

                persistent_directory = os.path.join("db",f"chroma_{extract_repo_name(repo_url)}_{chunk_size}_{chunk_overlap}_{embedding_model}_{llm}",)
                if not os.path.exists(persistent_directory):
                    create_vector_store(
                        data_path,
                        extensions,
                        persistent_directory,
                        chunk_size,
                        chunk_overlap,
                        embedding_model,
                        llm_summary,
                        llm,
                    )
                else:
                    console.print(
                        "[bold yellow]Vector store with those settings already exists.[/bold yellow]"
                    )

            except KeyboardInterrupt:
                console.print(
                    "\n[bold red]Indexing process interrupted by user.[/bold red]"
                )

        # List command
        elif user_input.lower() == "list":
            repo_dict = defaultdict(list)

            # Group repositories by username
            for folder in os.listdir("data"):
                user_path = os.path.join("data", folder)
                if os.path.isdir(user_path):
                    repo_dict[folder] = [
                        file for file in os.listdir(user_path)]

            # Display grouped repositories
            for user, repos in repo_dict.items():
                file_list = "\n".join([f"• {repo}" for repo in repos])
                panel_content = Text(file_list, style="bold yellow")
                console.print(
                    Panel(
                        panel_content,
                        title=f"[bold green]{user}'s Repositories[/bold green]",
                        expand=True,
                    ))

        # Clear command
        elif user_input.lower() == "clear":
            clear_screen()
            continue

        # Query command
        elif user_input.lower() == "query":
            # Allow the user to choose which vector store to query
            chroma_dict = defaultdict(list)

            for folder in os.listdir("db"):
                user_path = os.path.join("db", folder)
                if os.path.isdir(user_path):
                    chroma_dict[folder] = [
                        file for file in os.listdir(user_path)]

            # Flatten the list with indices
            chroma_list = []
            for user, chromas in chroma_dict.items():
                for chroma in chromas:
                    chroma_list.append((user, chroma))

            if not chroma_list:
                console.print("[bold red]No vector stores found![/bold red]")
                return

            # Display indexed options
            for i, (user, chroma) in enumerate(chroma_list, start=1):
                console.print(
                    f"[bold cyan]{i}[/bold cyan]: [bold green]{user}[/bold green] → {chroma}")

            # Prompt user for selection
            try:
                choice = int(
                    get_input_with_cancel(
                        "[bold blue]Enter the number of the vector store you want to query[/bold blue]"
                    )
                )
                if 1 <= choice <= len(chroma_list):
                    selected_user, selected_chroma = chroma_list[choice - 1]

                    # Start querying the chroma
                    persistent_directory = os.path.join(
                        "db", selected_user, selected_chroma
                    )
                    try:
                        start_query_ui(persistent_directory, args)
                        clear_screen()
                    except KeyboardInterrupt:
                        clear_screen()
                else:
                    console.print(
                        "[bold red]Error: Invalid selection.[/bold red]")
            except ValueError:
                console.print(
                    "[bold red]Error: Please enter a valid number.[/bold red]"
                )
            continue

        # Help command
        elif user_input.lower() == "help":
            console.print("\n[bold cyan]Available Commands:[/bold cyan]")
            console.print(
                "  [bold yellow]index[/bold yellow] - Index a GitHub repo")
            console.print(
                "  [bold yellow]query[/bold yellow] - Start querying over the GitHub repo"
            )
            console.print(
                "  [bold yellow]list[/bold yellow]  - Get a list of installed GitHub repos"
            )
            console.print(
                "  [bold yellow]exit[/bold yellow]  - Quit the assistant")
            console.print(
                "  [bold yellow]clear[/bold yellow] - Clear the terminal screen"
            )
            console.print(
                "  [bold yellow]help[/bold yellow]  - Show available commands"
            )
            continue
