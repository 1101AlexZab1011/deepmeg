import contextlib
import sys
from tqdm import tqdm
from typing import Generator, TextIO


class DummyFile(object):
    """
    A class that wraps a file object, replacing the 'write' method to avoid print() second call (useless \\n).
    """
    file: TextIO
    def __init__(self, file: TextIO):
        """
        Initialize the DummyFile object.

        Args:
            - file (TextIO) : The file object to be wrapped.
        """
        self.file = file

    def write(self, x: str):
        """
        Writes the given string to the wrapped file object, avoiding print() second call (useless \\n).

        Args:
            - x (str) : the string to be written.
        """
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        self.file.flush()

@contextlib.contextmanager
def nostdout() -> Generator:
    """
    A context manager that temporarily redirects the stdout to a DummyFile object.
    """
    if not isinstance(sys.stdout, DummyFile):
        save_stdout = sys.stdout
    else:
        save_stdout = sys.stdout.file

    sys.stdout = DummyFile(save_stdout)
    yield
    sys.stdout = save_stdout