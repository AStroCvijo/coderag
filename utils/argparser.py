import argparse

# Function for parsing arguments
def arg_parse():
    parser = argparse.ArgumentParser() 

    parser.add_argument('-m',  '--model',          type=str,            choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-3.5-turbo') 

    parser.add_argument('-ru', '--repo_url',       type=str,            default="https://github.com/viarotel-org/escrcpy")

    parser.add_argument('-ui', '--user_interface', action="store_true", default=False)
    parser.add_argument('-q',  '--query',          action="store_true", default=False)
    parser.add_argument('-e',  '--eval',           action="store_true", default=False)
    parser.add_argument('-v',  '--verbose',        action="store_true", default=False)

    parser.add_argument('-cs', '--chunk_size',     type=int,            default=1000)
    parser.add_argument('-co', '--chunk_overlap',  type=int,            default=100)
    parser.add_argument('-st', '--search_type',    type=str,            choices = ['similarity', 'similarity_score_threshold','mmr' ], default='similarity')    

    # Parse the arguments
    return parser.parse_args()