import argparse
import time

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


def edges_detection(img_path, square_x, square_y):
    max_sum = 0
    mean_sum = 0
    root_mean_square_sum = 0
    standard_deviation_sum = 0

    inImg = Image.open(img_path)
    #img must be grayscaled
    img_gray = rgb2gray(inImg)

    csvfile = open('SI_square.csv',mode='w',newline='')
    fieldnames = ['x', 'y', 'max', 'mean', 'root_mean_square', 'standard_deviation']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for x in range(0, img_gray.shape[1]+1-square_x, 10):
        for y in range(0, img_gray.shape[0]+1-square_y, 10):
            img_square=img_gray[y:y+square_y, x:x+square_x]

            #partial derivative of x_axis
            dx = ndimage.sobel(img_square,1)
            dy = ndimage.sobel(img_square,0)
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

            write.writerow({'x':x, 'y':y, 'max':max, 'mean':mean, 'root_mean_square':root_mean_square, 'standard_deviation':standard_deviation})

# main函数内使用
def edges_detection_square(img_path, square_x, square_y, step):
    inImg = Image.open(img_path)
    # img must be grayscaled
    img_gray = rgb2gray(inImg)

    list = []
    start_time = time.time()

    for x in range(0, img_gray.shape[1]+1-square_x, step):
        for y in range(0, img_gray.shape[0]+1-square_y, step):
            img_square=img_gray[y:y+square_y, x:x+square_x]

            # partial derivative of x_axis
            dx = ndimage.sobel(img_square, 1)
            dy = ndimage.sobel(img_square, 0)
            # magnitude
            mag = np.hypot(dx, dy)
            max = np.max(mag)
            mean = np.mean(mag)
            root_mean_square = math.sqrt(np.mean(np.power(mag, 2)))
            standard_deviation = math.sqrt(np.mean(np.power(mag, 2)) - np.power(mean, 2))

            list.append([x, y, max, mean, root_mean_square, standard_deviation])

    end_time = time.time()

    return list, end_time - start_time

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
    parser.add_argument('-input', type=str, default = './input.jpg')
    parser.add_argument('-x', type=int, default = 80)
    parser.add_argument('-y', type=int, default = 40)
    args = parser.parse_args()
    edges_detection(args.input, args.x, args.y)
    # input_img = cv2.imread(args.input)
    # output_img = edges_detection(input_img)
    # plot_res(input_img, output_img)
