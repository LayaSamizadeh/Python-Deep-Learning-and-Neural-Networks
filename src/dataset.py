import git
import os

def download_dataset(dataset_url, project_directory):
    # Clone the git repository
    repo = git.Repo.clone_from(dataset_url, project_directory)

    # Remove irrelevant files
    irrelevant_files = ['README.md', 'LICENSE']
    for filename in irrelevant_files:
        filepath = os.path.join(project_directory, filename)
        if os.path.exists(filepath):
            os.remove(filepath)