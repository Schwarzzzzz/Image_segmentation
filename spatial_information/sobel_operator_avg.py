import argparse
import cv2
import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy import ndimage
from PIL import Image
from statistics import mean


def edges_detection(img_path):
    paths = os.listdir(img_path)

    max_sum = 0
    mean_sum = 0
    root_mean_square_sum = 0
    standard_deviation_sum = 0
    i = 0

    for path in paths:
        i+=1
        img = img_path + '\\' + path
        inImg = Image.open(img)
        #img must be grayscaled
        img_gray = rgb2gray(inImg)

        #partial derivative of x_axis
        dx = ndimage.sobel(img_gray,1)
        dy = ndimage.sobel(img_gray,0)
        #magnitude
        mag = np.hypot(dx,dy)
        max = np.max(mag)
        mean = np.mean(mag)
        root_mean_square = math.sqrt(np.mean(np.power(mag, 2)))
        standard_deviation = math.sqrt(np.mean(np.power(mag, 2))-np.power(mean, 2))

        max_sum+=max
        mean_sum+=mean
        root_mean_square_sum+=root_mean_square
        standard_deviation_sum+=standard_deviation

        SI_avg = [max_sum/i, mean_sum/i, root_mean_square_sum/i, standard_deviation_sum/i]
        with open('SI_avg.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(SI_avg)

        #shown as image. must be 8 bit integer
        # mag = 255.0/ np.amax(mag)
        # mag = mag.astype(np.int8)

def plot_res(img,mag):
    figure = plt.figure()
    figure.add_subplot(211)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img)
        # plt.show()
    figure.add_subplot(212)
    plt.xticks([]), plt.yticks([])
    plt.imshow(mag)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default = './input')
    args = parser.parse_args()
    edges_detection(args.input)
    # input_img = cv2.imread(args.input)
    # output_img = edges_detection(input_img)
    # plot_res(input_img, output_img)
