from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import os

KERNELY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNELX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def load_image(path):
    """
    Load an image from the given path and return a numpy array.
    """
    if not os.path.exists(path):
        raise ValueError(f"Image at path '{path}' not found.")

    image = np.array(Image.open(path))

    # removing noise resulted in worse performance so I removed it

    return image

def edge_detection(image, save_path='edges.png'):
    """
    Perform edge detection using Sobel operator.
    Args:
        image: numpy array of the image.
        save_path: Path to save the resulting edge-detected image.

    Returns:
        Numpy array of the edge-detected image.
    """
    # Convert to grayscale if RGB
    if image.ndim == 3:
        bw_image = image.mean(axis=2)
    else:
        bw_image = image

    # Apply Sobel operator
    edgeX = convolve2d(bw_image, KERNELX, mode='same', boundary='symm')
    edgeY = convolve2d(bw_image, KERNELY, mode='same', boundary='symm')
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    # Normalize to 0-255
    edgeMAG = (edgeMAG / edgeMAG.max() * 255).astype(np.uint8)

    # Save if a path is provided
    if save_path:
        Image.fromarray(edgeMAG).save(save_path)

    return edgeMAG