
"""
@Author: Rezwan

"""
'''

* How many character number does SynthText in the Wild Dataset have? (https://stackoverflow.com/questions/63405973/how-many-character-number-does-synthtext-in-the-wild-dataset-have)

'''

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
import scipy.io as sio
import sys
import logging
import time
import json
import logging.config
import re
import argparse

def get_characters(basedir, imagedirname='SynthText', skip_missing=False):

    class Symbols:
        def __init__(self):
            self.symbols = set()

        def update(self, data):
            self.symbols = self.symbols.union(data)

        def __len__(self):
            return len(self.symbols)

        def __str__(self):
            return ''.join(self.symbols)

    symbols = Symbols()

    def csvgenerator(annodir, imagedir, cbb, wBB, imname, txts, symbols, **kwargs):
        image_num = kwargs.get('image_num')
        i = kwargs.get('i')

        imgpath = os.path.join(imagedir, imname)

        img = cv2.imread(imgpath)
        h, w, _ = img.shape
        if not os.path.exists(imgpath):
            if not skip_missing:
                raise FileNotFoundError('{} was not found'.format(imgpath))
            else:
                logging.warning('Missing image: {}'.format(imgpath))
                raise _Skip()


        # convert txts to list of str
        # I don't know why txts is
        # ['Lines:\nI lost\nKevin ', 'will                ', 'line\nand            ',
        # 'and\nthe             ', '(and                ', 'the\nout             ',
        # 'you                 ', "don't\n pkg          "]
        # there is strange blank and the length of txts is different from the one of wBB
        txts = ' '.join(txts.tolist()).split()
        text_num = len(txts)

        if wBB.ndim == 2:
            # convert shape=(2, 4,) to (2, 4, 1)
            wBB = np.expand_dims(wBB, 2)

        assert text_num == wBB.shape[2], 'The length of text and wordBB must be same, but got {} and {}'.format(
            text_num, wBB.shape[2])

        # replace non-alphanumeric characters with *
        alltexts_asterisk = ''.join([re.sub(r'[^A-Za-z0-9]', '*', text) for text in txts])
        assert len(alltexts_asterisk) == cbb.shape[
            2], 'The length of characters and cbb must be same, but got {} and {}'.format(
            len(alltexts_asterisk), cbb.shape[2])
        for b in tqdm(range(text_num)):
            text = txts[b]

            symboltext = re.sub(r'[A-Za-z0-9]+', '', text)

            symbols.update(symboltext)

        sys.stdout.write('\r{}, and number is {}...{:0.1f}% ({}/{})'.format(symbols, len(symbols), 100 * (float(i + 1) / image_num), i + 1, image_num))
        sys.stdout.flush()

    _gtmatRecognizer(csvgenerator, basedir, imagedirname, customLog=True, symbols=symbols)

    print()
    print('symbols are {}, and number is {}'.format(symbols, len(symbols)))


def _gtmatRecognizer(generator, basedir, imagedirname='SynthText', customLog=False, **kwargs):
    """
        convert gt.mat to https://github.com/MhLiao/TextBoxes_plusplus/blob/master/data/example.xml

        <annotation>
            <folder>train_images</folder>
            <filename>img_10.jpg</filename>
            <size>
                <width>1280</width>
                <height>720</height>
                <depth>3</depth>
            </size>
            <object>
                <difficult>1</difficult>
                <content>###</content>
                <name>text</name>
                <bndbox>
                    <x1>1011</x1>
                    <y1>157</y1>
                    <x2>1079</x2>
                    <y2>160</y2>
                    <x3>1076</x3>
                    <y3>173</y3>
                    <x4>1011</x4>
                    <y4>170</y4>
                    <xmin>1011</xmin>
                    <ymin>157</ymin>
                    <xmax>1079</xmax>
                    <ymax>173</ymax>
                </bndbox>
            </object>
            .
            .
            .

        </annotation>

        :param basedir: str, directory path under \'SynthText\'(, \'licence.txt\')
        :param imagedirname: (Optional) str, image directory name including \'gt.mat\
        :return:
        """
    logging.basicConfig(level=logging.INFO)

    imagedir = os.path.join(basedir, imagedirname)
    gtpath = os.path.join(imagedir, 'gt.mat')

    annodir = os.path.join(basedir, 'Annotations')

    if not os.path.exists(gtpath):
        raise FileNotFoundError('{} was not found'.format(gtpath))

    if not os.path.exists(annodir):
        # create Annotations directory
        os.mkdir(annodir)

    """
    ref: http://www.robots.ox.ac.uk/~vgg/data/scenetext/readme.txt
    gts = dict;
        __header__: bytes
        __version__: str
        __globals__: list
        charBB: object ndarray, shape = (1, image num). 
                Character level bounding box. shape = (2=(x,y), 4=(top left,...: clockwise), BBox word num)
        wordBB: object ndarray, shape = (1, image num). 
                Word level bounding box. shape = (2=(x,y), 4=(top left,...: clockwise), BBox char num)
        imnames: object ndarray, shape = (1, image num, 1).
        txt: object ndarray, shape = (i, image num).
             Text. shape = (word num)
    """
    logging.info('Loading {} now.\nIt may take a while.'.format(gtpath))
    gts = sio.loadmat(gtpath)
    logging.info('Loaded\n'.format(gtpath))

    charBB = gts['charBB'][0]
    wordBB = gts['wordBB'][0]
    imnames = gts['imnames'][0]
    texts = gts['txt'][0]

    image_num = imnames.size

    for i, (cbb, wBB, imname, txts) in tqdm(enumerate(zip(charBB, wordBB, imnames, texts))):
        imname = imname[0]

        try:
            generator(annodir, imagedir, cbb, wBB, imname, txts, i=i, image_num=image_num, **kwargs)
        except _Skip:
            pass

        if not customLog:
            sys.stdout.write('\rGenerating... {:0.1f}% ({}/{})'.format(100 * (float(i + 1) / image_num), i + 1, image_num))
        sys.stdout.flush()


    print()
    logging.info('Finished!!!')
    
  

  
if __name__ == '__main__':
    '''
        parsing and execution
    '''
    
    parser = argparse.ArgumentParser(description='Count characters of SynthText')
    parser.add_argument('synthtext', default=None, type=str,
                            help='path for synthtext dataset')
    args = parser.parse_args()
        
    time1 = time.time()
    
    #get_characters('./SynthText/SynthText/')
    get_characters(args.synthtext) ## Call function
    
    print('| Time taken for data init %.2f' % (time.time() - time1))

    

    
