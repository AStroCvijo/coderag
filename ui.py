import os
import getpass

from rag import *

from utils.argparser import arg_parse
from utils.const import extensions
from utils.repo import *

from eval.eval import eval

from user_interface.main_ui import start_ui

# Get the OpenAI API
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if __name__ == "__main__":

    # Parse the arguments
    args = arg_parse()

    # Start UI
    start_ui(args)
