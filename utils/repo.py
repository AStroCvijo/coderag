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