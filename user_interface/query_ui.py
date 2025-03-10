import os
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from rich.spinner import Spinner

from rag import query_vector_store
from utils.repo import parse_file_path

console = Console()

# Function for displaying the banner
def print_banner(name, repo):
    console.print(
        Panel(
            f"[bold green]Querying {name}'s {repo} repository![/bold green]", 
            border_style="cyan"
        )
    )

# Function for clearing the screen
def clear_screen(persistent_directory):
    os.system('cls' if os.name == 'nt' else 'clear')
    name, repo = parse_file_path(persistent_directory)
    print_banner(name, repo)

# Function to start the query UI
def start_query_ui(persistent_directory, args):
    clear_screen(persistent_directory)
    while True:
        # Get user input
        user_input = Prompt.ask("[bold blue]Enter query[/bold blue]")

        # Exit/Quit command
        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold red]Stopping the querying process![/bold red]")
            break

        # Clear command
        elif user_input.lower() == "clear":
            clear_screen()
            continue

        # Help command
        elif user_input.lower() == "help":
            console.print("\n[bold cyan]Available Commands:[/bold cyan]")
            console.print("  [bold yellow]exit[/bold yellow]  - Quit the assistant")
            console.print("  [bold yellow]clear[/bold yellow] - Clear the terminal screen")
            console.print("  [bold yellow]help[/bold yellow]  - Show available commands\n")
            continue

        # Else query user input
        else:
            # Query the vector store
            with console.status("[bold green]Querying vector store...[/bold green]", spinner="dots"):
                retrieved_files = query_vector_store(user_input, persistent_directory, args.k, args.embedding_model)
                
                if retrieved_files:
                    file_list = "\n".join([f"• {file}" for file in retrieved_files])
                    panel_content = Text(file_list, style="bold yellow")
                    console.print(Panel(panel_content, title="[bold green]Retrieved Files[/bold green]", expand=False))
                else:
                    console.print("[bold red]No files found.[/bold red]")
