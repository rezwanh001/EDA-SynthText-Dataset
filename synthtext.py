
"""
@Author: Rezwan

"""
'''

* Source Link: https://gist.github.com/soumith/a9c86506928e33fabcd4d752cc1f50c7

'''

import numpy as np
import os
import time
import warnings
import pickle
import argparse
import time
import json
import logging.config
import re
from scipy import misc
from tqdm import tqdm
# from accimage import Image
from PIL import Image
import io

try:
    from fasterzip import ZipFile
    fastzip = True
except:
    warnings.warn('For faster loading of zip, you can install fasterzip via '
                  '`pip install https://github.com/TkTech/fasterzip/archive/master.zip`')
    from zipfile import ZipFile
    fastzip = False

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import scipy.io

def preprocess_words(word_ar):
    words = []
    for ii in range(np.shape(word_ar)[0]):
        s = word_ar[ii]
        start = 0
        while s[start] == ' ' or s[start] == '\n':
            start += 1
        for i in range(start + 1, len(s) + 1):
            if i == len(s) or s[i] == '\n' or s[i] == ' ':
                if start != i:
                    words.append(s[start : i])
                start = i + 1
    return words

def get_path(path):
    if fastzip:
        return path.encode()
    else:
        return path

class SynthTextDataset(Dataset):
    def __init__(self, zip_path, cache_path=None):
        self.zip_path = zip_path
        self.cache_path = cache_path

    def lazy_init(self):
        """
        we lazily initialize rather than opening the Zip file before-hand because,
        ZipFile is not thread/fork safe. If we dont lazily initialize,
        then a bunch of file read errors show up, that are false-positives (the zip itself is fine)
        """
        if hasattr(self, 'images'):
            return
        zip_path = self.zip_path
        cache_path = self.cache_path
        tm = time.time()
        self.zip = ZipFile(get_path(zip_path))
        print('opening zip file....done in {} seconds'.format(time.time() - tm))
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(zip_path), 'gt.pkl')
        tm = time.time()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                gt = pickle.load(f)
        else:
            if fastzip:
                with self.zip.read(get_path(os.path.join('SynthText', 'gt.mat'))) as file_contents:
                    gt_ = scipy.io.loadmat(io.BytesIO(file_contents))
            else:
                file_contents = self.zip.read(get_path(os.path.join('SynthText', 'gt.mat')))
                gt_ = scipy.io.loadmat(io.BytesIO(file_contents))
            gt = {
                'imnames': gt_['imnames'],
                'wordBB' : gt_['wordBB'],
                'txt' : gt_['txt'],
            }
            del gt_
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(gt, f, protocol=pickle.HIGHEST_PROTOCOL)
            except e:
                warnings.warn("Couldn't write SynthTextDataset cache at {}".format(cache_path))
        self.images = gt['imnames'][0]
        self.bboxes = gt['wordBB'][0]
        self.text = gt['txt'][0]
        del gt
        print('loading metadata done in {} seconds'.format(time.time() - tm))

    def __len__(self):
        return 858750 # size of SynthText dataset

    def __getitem__(self, index):
        self.lazy_init()
        path = str(self.images[index][0])
        boxes = self.bboxes[index]
        words = preprocess_words(self.text[index])
        im = 'SynthText/' + path
        if len(np.shape(boxes)) == 2:
            boxes = np.array([boxes])
            boxes = np.transpose(boxes, (1, 2, 0))

        boxes = np.transpose(boxes, (2, 1, 0)) # num_boxes, 4 points, 2 xy
        try:
            if fastzip:
                with self.zip.read(get_path(im)) as imbytes:
                    # pil_img = Image(bytes(imbytes))
                    pil_img = Image.open(io.BytesIO(bytes(imbytes)))
            else:
                imbytes = self.zip.read(get_path(im))
                pil_img = Image.open(io.BytesIO(imbytes))
                # pil_img = Image(bytes(imbytes))
        except Exception as e:
            print(e, index, path)
            raise e

        return pil_img, boxes, words


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('synthtext', default=None, type=str,
                        help='path for synthtext dataset')
    args = parser.parse_args()

    time1 = time.time()

    dataset = SynthTextDataset(args.synthtext) # Path = "../SynthText.zip"
    print('| Time taken for data init %.2f' % (time.time() - time1))
    sample = dataset[0]
    print(sample)