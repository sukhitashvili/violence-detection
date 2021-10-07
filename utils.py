from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def plot(image: np.array, title: str, title_size: int = 16,
         figsize: tuple = (13, 7),
         save_path: Union[str, None] = None):
    plt.figure(figsize=figsize)
    plt.title(title, size=title_size)
    plt.axis('off')
    plt.imshow(image)
    if save_path:
        plt.savefig(save_path)
