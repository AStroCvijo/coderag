import os
import sys
import readline
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.text import Text
from rag import query_vector_store

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

# Function to start the query UI
def start_query_ui(persistent_directory):
    clear_screen()
    while True:
        user_input = Prompt.ask("[bold blue]Enter query[/bold blue]")

        if user_input.lower() in {"exit", "quit"}:
            console.print("\n[bold red]Goodbye![/bold red] 👋")
            break
        elif user_input.lower() == "clear":
            clear_screen()
            continue
        elif user_input.lower() == "help":
            console.print("\n[bold cyan]Available Commands:[/bold cyan]")
            console.print("  [bold yellow]exit[/bold yellow]  - Quit the assistant")
            console.print("  [bold yellow]clear[/bold yellow] - Clear the terminal screen")
            console.print("  [bold yellow]help[/bold yellow]  - Show available commands\n")
            continue
        else:
            # Query the vector store
            with console.status("[bold green]Querying vector store...[/bold green]", spinner="dots"):
                retrieved_files = query_vector_store(user_input, persistent_directory)
                
                if retrieved_files:
                    file_list = "\n".join([f"• {file}" for file in retrieved_files])
                    panel_content = Text(file_list, style="bold yellow")
                    console.print(Panel(panel_content, title="[bold green]Retrieved Files[/bold green]", expand=False))
                else:
                    console.print("[bold red]No files found.[/bold red]")
