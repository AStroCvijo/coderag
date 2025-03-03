import argparse

# Function for parsing arguments
def arg_parse():
    parser = argparse.ArgumentParser() 

    parser.add_argument('-m',  '--model',       type=str,            choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-3.5-turbo') 

    parser.add_argument('-ru', '--repo_url',    type=str,            default="https://github.com/viarotel-org/escrcpy")

    parser.add_argument('-q',  '--query',       action="store_true", default=False)
    parser.add_argument('-e',  '--eval',        action="store_true", default=False)
    parser.add_argument('-v',  '--verbose',     action="store_true", default=False)
    parser.add_argument('-st', '--search_type', type=str,            choices = ['similarity'], default='similarity')    

    # Parse the arguments
    return parser.parse_args()