from PIL import Image
import numpy as np
from scipy.ndimage import convolve
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

def edge_detection(image_array):
    gray_image = np.mean(image_array, axis=2)
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edgeY = convolve(gray_image, kernelY, mode='constant', cval=0.0)
    edgeX = convolve(gray_image, kernelX, mode='constant', cval=0.0)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG