import os
import sys
import readline
from collections import defaultdict
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text

from user_interface.query_ui import start_query_ui

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
def start_ui(persistent_directory, args):
    clear_screen()
    
    while(True):
        user_input = Prompt.ask("[bold blue]Enter command[/bold blue]")
        
        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold red]Goodbye![/bold red] 👋")
            break
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
            start_query_ui(persistent_directory, args)
            clear_screen()
            continue
        elif user_input.lower() == "help":
            console.print("\n[bold cyan]Available Commands:[/bold cyan]")
            console.print("  [bold yellow]list[/bold yellow]  - Get a list of installed GitHub repos")
            console.print("  [bold yellow]query[/bold yellow]  - Start querying over the DataBase")
            console.print("  [bold yellow]exit[/bold yellow]  - Quit the assistant")
            console.print("  [bold yellow]clear[/bold yellow] - Clear the terminal screen")
            console.print("  [bold yellow]help[/bold yellow]  - Show available commands\n")
            continue