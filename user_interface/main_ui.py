import os
import sys
import readline
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text
import validators

from rag import create_vector_store
from utils.repo import *
from user_interface.query_ui import start_query_ui
from utils.const import extensions
from eval.eval import eval

console = Console()

# Function for displaying the banner
def print_banner():
    console.print(
        Panel(
            "[bold cyan]Welcome to the AI-Powered RAG System![/bold cyan]", 
            border_style="cyan"
        )
    )

# Function for clearing the screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()

def get_input_with_cancel(prompt_text):
    user_input = Prompt.ask(prompt_text)
    if user_input.lower() == "cancel":
        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
        return None
    return user_input


# Function to start the UI
def start_ui():
    clear_screen()
    
    while(True):
        user_input = Prompt.ask("[bold blue]Enter command[/bold blue]")
        
        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold red]Goodbye![/bold red] 👋")
            break

        elif user_input.lower() == "index":
            try:
                # Keep prompting until a valid GitHub URL is entered
                while True:
                    repo_url = get_input_with_cancel("[bold blue]Enter GitHub repository URL[/bold blue]")
                    if repo_url is None:
                        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
                        return
                    if validators.url(repo_url) and "github.com" in repo_url:
                        break
                    console.print("[bold red]Error: The provided URL is not a valid GitHub repository link.[/bold red]")

                data_path = os.path.join("data", extract_repo_name(repo_url))
                clone_repo(repo_url, data_path)

                # Keep prompting until a valid chunk size is entered
                while True:
                    chunk_size = get_input_with_cancel("[bold blue]Enter chunk size (positive integer)[/bold blue]")
                    if chunk_size is None:
                        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
                        return
                    if chunk_size.isdigit() and int(chunk_size) > 0:
                        chunk_size = int(chunk_size)
                        break
                    console.print("[bold red]Error: Chunk size must be a positive integer.[/bold red]")

                # Keep prompting until a valid chunk overlap is entered
                while True:
                    chunk_overlap = get_input_with_cancel("[bold blue]Enter chunk overlap (non-negative integer)[/bold blue]")
                    if chunk_overlap is None:
                        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
                        return
                    if chunk_overlap.isdigit() and int(chunk_overlap) >= 0:
                        chunk_overlap = int(chunk_overlap)
                        break
                    console.print("[bold red]Error: Chunk overlap must be a non-negative integer.[/bold red]")

                # Keep prompting until a valid LLM summary option is entered
                while True:
                    llm_summary = get_input_with_cancel("[bold blue]Use LLM summary? (Y/n):[/bold blue]")
                    if llm_summary is None:
                        console.print("[bold yellow]Indexing canceled by user.[/bold yellow]")
                        return
                    if llm_summary.lower() in ["y", "n", "yes", "no"]:
                        break
                    console.print("[bold red]Error: Please enter 'Y' or 'N' for LLM summary.[/bold red]")

                persistent_directory = os.path.join("db", f"chroma_{extract_repo_name(repo_url)}_{chunk_size}_{chunk_overlap}")
                if not os.path.exists(persistent_directory):
                    create_vector_store(data_path, extensions, persistent_directory, chunk_size, chunk_overlap)
                else:
                    console.print("[bold yellow]Vector store already exists.[/bold yellow]")

            except KeyboardInterrupt:
                console.print("\n[bold red]Indexing process interrupted by user.[/bold red]")


        elif user_input.lower() == "list":
            repo_dict = defaultdict(list)        
            
            # Group repositories by username
            for folder in os.listdir('data'):
                user_path = os.path.join('data', folder)
                if os.path.isdir(user_path):
                    repo_dict[folder] = [file for file in os.listdir(user_path)]
            
            # Display grouped repositories
            for user, repos in repo_dict.items():
                file_list = "\n".join([f"• {repo}" for repo in repos])
                panel_content = Text(file_list, style="bold yellow")
                console.print(Panel(panel_content, title=f"[bold green]{user}'s Repositories[/bold green]", expand=True))

        elif user_input.lower() == "clear":
            clear_screen()
            continue
        elif user_input.lower() == "query":
            # Allow the user to choose which vector store to query
            chroma_dict = defaultdict(list)
            
            for folder in os.listdir('db'):
                user_path = os.path.join('db', folder)
                if os.path.isdir(user_path):
                    chroma_dict[folder] = [file for file in os.listdir(user_path)]
            
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
                console.print(f"[bold cyan]{i}[/bold cyan]: [bold green]{user}[/bold green] → {chroma}")

            # Prompt user for selection
            try:
                choice = int(input("\nEnter the number of the vector store you want to query: "))
                if 1 <= choice <= len(chroma_list):
                    selected_user, selected_chroma = chroma_list[choice - 1]
                    console.print(f"\n[bold yellow]You selected:[/bold yellow] {selected_user} → {selected_chroma}")
                    
                    # Start querying the chroma
                    persistent_directory = os.path.join('db', selected_user, selected_chroma)
                    try:
                        start_query_ui(persistent_directory)
                    except KeyboardInterrupt:
                        clear_screen()
                else:
                    console.print("[bold red]Error: Invalid selection.[/bold red]")
            except ValueError:
                console.print("[bold red]Error: Please enter a valid number.[/bold red]")
            continue

        elif user_input.lower() == "help":
            console.print("\n[bold cyan]Available Commands:[/bold cyan]")
            console.print("  [bold yellow]index[/bold yellow] - Index a GitHub repo")
            console.print("  [bold yellow]query[/bold yellow] - Start querying over the GitHub repo")
            console.print("  [bold yellow]list[/bold yellow]  - Get a list of installed GitHub repos")
            console.print("  [bold yellow]exit[/bold yellow]  - Quit the assistant")
            console.print("  [bold yellow]clear[/bold yellow] - Clear the terminal screen")
            console.print("  [bold yellow]help[/bold yellow]  - Show available commands\n")
            continue