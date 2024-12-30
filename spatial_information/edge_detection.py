import argparse
import cv2
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import rgb2gray
from scipy import ndimage
from PIL import Image
from statistics import mean


def edges_detection(img_path):
    img = img_path
    inImg = Image.open(img)
    #img must be grayscaled
    img_gray = rgb2gray(inImg)

    #partial derivative of x_axis
    dx = ndimage.sobel(img_gray,1)
    dy = ndimage.sobel(img_gray,0)
    #magnitude
    mag = np.hypot(dx,dy)
    return mag

if __name__ == "__main__":
    input_img = cv2.imread('input.jpg')
    output_img = edges_detection('input.jpg')
    figure = plt.figure()
    plt.xticks([]), plt.yticks([])
    plt.imshow(output_img)
    plt.show()