from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot(image: np.array, title: str, title_size: int = 16,
         figsize: tuple = (13, 7),
         convert_to_rgb: bool = True,
         save_path: Union[str, None] = None):
    if convert_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.title(title, size=title_size)
    plt.axis('off')
    plt.imshow(image)
    if save_path:
        plt.savefig(save_path)
