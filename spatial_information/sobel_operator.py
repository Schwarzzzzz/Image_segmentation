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
    global output_array
    paths = os.listdir(img_path)
    i=0

    csvfile = open('SI.csv',mode='w',newline='')
    fieldnames = ['i', 'max', 'mean', 'root_mean_square', 'standard_deviation']
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
        max = np.max(mag)
        mean = np.mean(mag)
        root_mean_square = math.sqrt(np.mean(np.power(mag, 2)))
        standard_deviation = math.sqrt(np.mean(np.power(mag, 2))-np.power(mean, 2))
        write.writerow({'i':path, 'max':max, 'mean':mean, 'root_mean_square':root_mean_square, 'standard_deviation':standard_deviation})
        # if i == 0:
            # output_array = np.array([[max, mean, root_mean_square, standard_deviation]])
        # else:
            # output_array = np.r_[output_array, ([[max, mean, root_mean_square, standard_deviation]])]
        # i = i + 1
    # return pd.DataFrame(output_array, columns=['max', 'mean', 'root_mean_square', 'standard_deviation'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default = './input')
    args = parser.parse_args()
    edges_detection(args.input)
    # input_img = cv2.imread(args.input)
    # output_img = edges_detection(input_img)
    # plot_res(input_img, output_img)



'''
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
    global output_array
    paths = ['os.listdir(img_path)']
    i=0

    csvfile = open('SI.csv',mode='w',newline='')
    fieldnames = ['i', 'max', 'mean', 'root_mean_square', 'standard_deviation']
    write = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for path in paths:
        img = 'input.jpg'
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
        write.writerow({'i':path, 'max':max, 'mean':mean, 'root_mean_square':root_mean_square, 'standard_deviation':standard_deviation})
        # if i == 0:
            # output_array = np.array([[max, mean, root_mean_square, standard_deviation]])
        # else:
            # output_array = np.r_[output_array, ([[max, mean, root_mean_square, standard_deviation]])]
        # i = i + 1
    # return pd.DataFrame(output_array, columns=['max', 'mean', 'root_mean_square', 'standard_deviation'])
    return mag

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
    # edges_detection(args.input)
    input_img = cv2.imread('input.jpg')
    output_img = edges_detection('input.jpg')
    plot_res(input_img, output_img)
'''