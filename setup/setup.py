import os

# Setup information.
workspace_name = 'workspace'

def setup_workspace():
    """Sets up the workspace for the project by creating a virtual environment,
    installing the required packages and creating symlinks for the dataset.
    """
    # Create the workspace folder if it does not exist.
    workspace_path = os.path.abspath(workspace_name)
    if not os.path.isdir(workspace_path):
        print('Creating workspace: {}'.format(workspace_path))
        os.makedirs(workspace_path)

if __name__ == '__main__':
    setup_workspace()