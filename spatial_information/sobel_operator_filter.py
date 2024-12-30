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


def edges_detection(img_path, threshold):
    global output_array
    paths = os.listdir(img_path)
    i=0

    csvfile = open('SI.csv',mode='w',newline='')
    fieldnames = ['i', 'percentage', 'max', 'mean', 'root_mean_square', 'standard_deviation']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for path in paths:
        img = img_path + '\\' + path
        inImg = Image.open(img)
        #img must be grayscaled
        img_gray = rgb2gray(inImg)

        #partial derivative of x_axis
        dx = ndimage.sobel(img_gray,1)
        dy = ndimage.sobel(img_gray,0)
        #magnitude
        mag = np.hypot(dx,dy)
        # filtered_mag = mag[mag > threshold]
        filtered_mag = mag
        filtered_mag[filtered_mag <= threshold] = 0
        percentage = np.sum(mag > threshold)/np.sum(mag > -1)
        max = np.max(filtered_mag)
        mean = np.mean(filtered_mag)
        root_mean_square = math.sqrt(np.mean(np.power(filtered_mag, 2)))
        standard_deviation = math.sqrt(np.mean(np.power(filtered_mag, 2))-np.power(mean, 2))
        write.writerow({'i':path, 'percentage':percentage, 'max':max, 'mean':mean, 'root_mean_square':root_mean_square, 'standard_deviation':standard_deviation})
    # return pd.DataFrame(output_array, columns=['max', 'mean', 'root_mean_square', 'standard_deviation'])

def cal_threshold(img_path):
    inImg = Image.open(img_path)
    #img must be grayscaled
    img_gray = rgb2gray(inImg)

    #partial derivative of x_axis
    dx = ndimage.sobel(img_gray,1)
    dy = ndimage.sobel(img_gray,0)
    #magnitude
    mag = np.hypot(dx,dy)
    threshold = np.percentile(mag, 90)
    mag[mag <= threshold] = 0
    return threshold, mag

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
    # 输入原图用于计算阈值
    parser.add_argument('-ori', type=str, default = './input.jpg')
    parser.add_argument('-input', type=str, default = './input')
    args = parser.parse_args()
    threshold, output_img = cal_threshold(args.ori)
    edges_detection(args.input, threshold)
    # input_img = cv2.imread(args.ori)
    plot_res(input_img, output_img)
