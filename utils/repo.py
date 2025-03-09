import re
import os
from git import Repo

# Function for extracting Github repo name
def extract_repo_name(url: str) -> str:
    match = re.search(r'github\.com/([^/]+/[^/]+)', url)
    return match.group(1)

# Function for cloning a GitHub repo
def clone_repo(repo_url, data_path):
    if not os.path.exists(data_path):
        print(f"Cloing the repo at {repo_url}...")
        Repo.clone_from(repo_url, data_path)
        print(f"Repository cloned to {data_path}.")
    else:
        print("Repo already cloned.")

def parse_file_path(file_path):
    pattern = r'chroma_([^/]+)/([^/]+)_\d{4}_\d{3}$'

    match = re.search(pattern, file_path)
    if match:
        github_user = match.group(1)
        repo_name = match.group(2)
        return github_user, repo_name
    else:
        raise ValueError("File path does not match the expected format")