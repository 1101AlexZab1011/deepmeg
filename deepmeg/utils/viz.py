import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


def generate_cmap(*colors: str) -> ListedColormap:
    """Generate a ListedColormap using the provided colors.

    This function generates a `ListedColormap` with a color gradient
    between each pair of adjacent colors in the `colors` list.

    Parameters
    ----------
    *colors : str
        List of hexadecimal color strings.

    Returns
    -------
    matplotlib.colors.ListedColormap
        ListedColormap with a gradient between the provided colors.

    Examples
    --------
    >>> generate_cmap("#FF0000", "#00FF00", "#0000FF")
    ListedColormap(['#ff0000', '#00ff00', '#0000ff'], N=256)

    """
    def crange(a: float, b: float, N: int) -> np.ndarray:
        """Return a range of numbers from a to b, with N values."""
        if a < b:
            return np.arange(a, b, (abs(a - b))/N)
        elif a > b:
            return np.arange(b, a, (abs(a - b))/N)[::-1]
        else:
            return a*np.ones(N)

    N = 256
    all_vals = list()
    for from_color, to_color in zip(colors[:-1], colors[1:]):
        vals = np.ones((N, 4))
        a1, a2, a3 = mcolors.hex2color(from_color)
        b1, b2, b3 = mcolors.hex2color(to_color)
        vals[:, 0] = crange(a1, b1, N)
        vals[:, 1] = crange(a2, b2, N)
        vals[:, 2] = crange(a3, b3, N)
        all_vals.append(vals)

    return ListedColormap(np.vstack(all_vals))