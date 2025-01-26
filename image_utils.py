from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.filters import median
from skimage.morphology import ball
import os

KERNELY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
KERNELX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def load_image(path):
    if not os.path.exists(path):
        raise ValueError(f"Image at path '{path}' not found.")
    return np.array(Image.open(path))

def preprocess_image(image):
    """
    Apply a median filter for noise reduction.
    """
    return median(image, ball(3))

def edge_detection(image, threshold=20, save_path='edges.png'):
    """
    Perform edge detection using Sobel operator and binary thresholding.
    """
    # apply median filter to reduce noise
    image = preprocess_image(image)
    if image.ndim == 3:
        bw_image = image.mean(axis=2)
    else:
        bw_image = image

    edgeX = convolve2d(bw_image, KERNELX, mode='same', boundary='symm')
    edgeY = convolve2d(bw_image, KERNELY, mode='same', boundary='symm')
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    edgeMAG = (edgeMAG / edgeMAG.max() * 255).astype(np.uint8)
    edge_binary = edgeMAG > threshold  # Binary threshold

    if save_path:
        Image.fromarray((edge_binary * 255).astype(np.uint8)).save(save_path)

    return edge_binary
