#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:16:32 2018

@author: linsam
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.data import imread
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
ship_dir = '/home/linsam/github/AirbusShipDetectionChallenge/'
train_image_dir = ship_dir + 'train/'
test_image_dir = ship_dir + 'test/'

train = os.listdir(train_image_dir)
print(len(train))
test = os.listdir(test_image_dir)
print(len(test))

submission = pd.read_csv(ship_dir+'sample_submission.csv')
submission.head()


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    #starts -= 1
    ends = starts + lengths -1 
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    
    for s, e in zip(starts, ends):
        img[s:e] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
    #return img.reshape(shape)  # Needed to align to RLE direction

masks = pd.read_csv( ship_dir + 'train_ship_segmentations.csv')
masks.head()

ImageId = '0005d01c8.jpg'

img = imread( train_image_dir + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

plt.imshow(img)
# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768))
for mask in img_masks:
    print(1)
    all_masks += rle_decode(mask)

img = imread( train_image_dir + ImageId)
np.array(img).shape
plt.imshow(img)
#img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
tem = masks[ masks.ImageId == ImageId ].EncodedPixels
#tem[tem.index[0]]
all_masks.shape
plt.imshow(all_masks)
plt.imshow( rle_decode(tem[tem.index[0]]) )
plt.imshow( rle_decode(tem[tem.index[1]]) )



fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
#axarr[0].axis('off')
#axarr[1].axis('off')
#axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

plt.imshow(all_masks)










