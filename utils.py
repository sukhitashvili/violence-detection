import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot(image: np.array, figsize: tuple = (13, 13),
         convert_to_rgb: bool = True):
    if convert_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(image)
