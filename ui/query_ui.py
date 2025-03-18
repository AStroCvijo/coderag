import os
from utils.const import *
from rich.text import Text
from rich.panel import Panel
from rich.prompt import Prompt
from rich.console import Console
from rag import query_vector_store
from rag import query_vector_store_with_llm
from utils.repo import parse_file_path

console = Console()


# Function for displaying the banner
def print_banner(name, repo):
    console.print(
        Panel(
            f"[bold green]Querying {name}'s {repo} repository![/bold green] ([italic yellow]Type HELP for a list of commands[/italic yellow])",
            border_style="cyan",
        )
    )


# Function for clearing the screen
def clear_screen(persistent_directory):
    os.system("cls" if os.name == "nt" else "clear")
    name, repo, _, _ = parse_file_path(persistent_directory)
    print_banner(name, repo)


# Function to start the query UI
def start_query_ui(persistent_directory, args):
    clear_screen(persistent_directory)
    _, _, embedding_model, _ = parse_file_path(persistent_directory)

    # Get available models and have user select one
    models = list(MODEL_CONTEXT_LENGTHS.keys())
    console.print(
        "\n[bold cyan]Select an LLM for generating summaries:[/bold cyan]")
    for idx, model in enumerate(models, 1):
        console.print(f"  {idx}. {model}")
    selected = Prompt.ask(
        "Enter the number of your choice",
        choices=[str(i) for i in range(1, len(models) + 1)],
        default="1",
    )
    llm = models[int(selected) - 1]

    while True:
        user_input = Prompt.ask("\n[bold blue]Enter query[/bold blue]")

        # Exit/Quit command
        if user_input.lower() in {"exit", "quit"}:
            console.print(
                "\n[bold red]Stopping the querying process![/bold red]")
            break

        # Clear command
        elif user_input.lower() == "clear":
            clear_screen(persistent_directory)
            continue

        # Help command
        elif user_input.lower() == "help":
            console.print("\n[bold cyan]Available Commands:[/bold cyan]")
            console.print(
                "  [bold yellow]exit[/bold yellow]  - Quit the assistant")
            console.print(
                "  [bold yellow]clear[/bold yellow] - Clear the terminal screen"
            )
            console.print(
                "  [bold yellow]help[/bold yellow]  - Show available commands"
            )
            console.print(
                "  [bold yellow]model[/bold yellow] - Change the LLM for summaries"
            )
            continue

        # Model selection command
        elif user_input.lower() == "model":
            console.print("\n[bold cyan]Available LLMs:[/bold cyan]")
            for idx, model in enumerate(models, 1):
                console.print(f"  {idx}. {model}")
            selected = Prompt.ask(
                "Enter the number of your choice",
                choices=[str(i) for i in range(1, len(models) + 1)],
                default="1",
            )
            llm = models[int(selected) - 1]
            console.print(f"[bold green]Switched to model: {llm}[/bold green]")
            continue

        # Query processing
        else:
            # Query the vector store
            with console.status(
                "[bold green]Querying vector store...[/bold green]", spinner="dots"
            ):
                retrieved_files = query_vector_store(
                    user_input, persistent_directory, args.top_k, embedding_model)

                if retrieved_files:
                    file_list = "\n".join(
                        [f"• {file}" for file in retrieved_files])
                    panel_content = Text(file_list, style="bold yellow")
                    console.print(
                        Panel(
                            panel_content,
                            title="[bold yellow]Retrieved Files[/bold yellow]",
                            expand=False,
                        )
                    )
                else:
                    console.print("[bold red]No files found.[/bold red]")

            # Generate an answer with the current LLM
            with console.status(
                "[bold green]Generating summary...[/bold green]", spinner="dots"
            ):
                answer = query_vector_store_with_llm(
                    user_input, persistent_directory, embedding_model, llm
                )
                console.print(f"[bold magenta]Answer: {answer}[/bold magenta]")
