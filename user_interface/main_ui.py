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

# Function to start the UI
def start_ui(args):
    clear_screen()
    
    while(True):
        user_input = Prompt.ask("[bold blue]Enter command[/bold blue]")
        
        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold red]Goodbye![/bold red] 👋")
            break

        elif user_input.lower() == "index":

            # Prompt the user for the GitHub repo url
            repo_url = Prompt.ask("[bold blue]Enter HitHub repository URL[/bold blue]")
            if not validators.url(repo_url) or "github.com" not in repo_url:
                console.print("[bold red]Invalid GitHub repository URL![/bold red]")
                continue

            # Clone the repository
            data_path = os.path.join("data", extract_repo_name(repo_url))
            clone_repo(repo_url, data_path)

            # Prompt the user for vector store params
            chunk_size = int(Prompt.ask("[bold blue]Enter chunk size[/bold blue]"))
            chunk_overlap = int(Prompt.ask("[bold blue]Enter chunk overlap[/bold blue]"))
            # to be implemented
            llm_summary = Prompt.ask("[bold blue]Use LLM summary [Y/n][/bold blue]")
                
            # Create the vector store
            persistent_directory = os.path.join("db", f"chroma_{extract_repo_name(repo_url)}_{chunk_size}_{chunk_overlap}")
            if not os.path.exists(persistent_directory):
                create_vector_store(data_path, extensions, persistent_directory, chunk_size, chunk_overlap)
            else:
                print("Vector store already exists.")

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
                    start_query_ui(persistent_directory, args)
                else:
                    console.print("[bold red]Invalid selection![/bold red]")
            except ValueError:
                console.print("[bold red]Please enter a valid number![/bold red]")
            
            clear_screen()
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