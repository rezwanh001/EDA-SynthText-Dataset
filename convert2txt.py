"""
@Author: Rezwan

"""


# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from scipy import misc
from tqdm import tqdm
import scipy.io
import re
import argparse
import sys
import logging
import time
import json

def convert2txt(path):
    """
    :param path:
    :return:
    """
    # This function loads the mat file and returns a dictionary
    data = scipy.io.loadmat(path)

    # # The absolute address of the image
    # abs_path = data["imnames"][0]
    # # Get location information of bbox (for all pictures)
    # coordinate = data["wordBB"][0]
    # # Get comments (characters, text labels)
    # string_anns = data["txt"][0]
    # print()
    # # Use the loop to get each image and its coordinate information and label
    # for i in  range(len(data['txt'][0])):
    # Need a loop to traverse all gt values
    train_file = open('./SynthText/SynthText/train.txt', 'w')
    print("Start writing")
    for i in tqdm(range(len(data['txt'][0]))):

        for val in tqdm(data['txt'][0][i]):
            # Remove line breaks and spaces
            v = [x.split("\n") for x in val.strip().split(" ")]

        # sys.stderr returns an error message
        # print(sys.stderr, "No.{} data".format(i))
        rec = np.array(data['wordBB'][0][i], dtype=np.int32)
        if len(rec.shape) == 3:
            rec = rec.transpose([2, 1, 0])
        else:
            rec = rec.transpose([1, 0])[np.newaxis, :]
        # Write the coordinate value of each rectangular box to the txt file
        for j in range(len(rec)):
            x1 = rec[j][0][0]
            y1 = rec[j][0][1]
            x2 = rec[j][1][0]
            y2 = rec[j][1][1]
            x3 = rec[j][2][0]
            y3 = rec[j][2][1]
            x4 = rec[j][3][0]
            y4 = rec[j][3][1]

            train_file.write(str(data['imnames'][0][i][0]) + "," + str(x1) + "," + str(y1) + "," + str(x2) + "," + str(
                y2) + "," + str(x3) + "," + str(y3) + "," + str(x4) + "," + str(y4)
                             + "," + str(v[0][:]) + "\n")

if __name__ == '__main__':
    '''
        parsing and execution
    '''
    
    parser = argparse.ArgumentParser(description='Convet to text data')
    parser.add_argument('synthtext_gt', default=None, type=str,
                            help='path for synthtext gt.mat file')
    args = parser.parse_args()
        
    time1 = time.time()
    
    # convert2txt("./SynthText/SynthText/SynthText/gt.mat")
    convert2txt(args.synthtext_gt) ## Call function
    
    print('| Time taken for data init %.2f' % (time.time() - time1))

