import os

def check_path(*args: str) -> None:
    """Creates a directory if does not exist and if it is possible.

    Args:
        *args (str): Directories to check.
    """
    for path in args:
        if not os.path.isdir(path):
            os.mkdir(path)
