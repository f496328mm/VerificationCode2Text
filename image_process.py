#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 01:52:24 2018

@author: linsam
"""
import numpy as np

def image2EncodedPixels(captcha):
    captcha = np.array(captcha)
    captcha = captcha[:,:,0]
    captcha2 = captcha.T.reshape((60*200))
    
    value = []
    for i in range(len(captcha2)):
        if captcha2[i] != 0 :
            value.append(i)
           
    value2 = ''
    bo = 0
    total = 0
    for i in range(len(value)):
        #print( 'i = ' + str( i ))
        #print( 'bo = ' + str( bo ))
        if bo == 0:
            value2 = value2 + str(value[i]) + ' '
            bo = 1
        elif bo == 1: 
            if value[i] - value[i-1] == 1:
                total = total + 1
                
            elif value[i] - value[i-1] > 1:
                value2 = value2 + str(total) + ' '
                bo = 0
                total = 0
        if i == (len(value)-1) :
            value2 = value2 + str(total)   
            
    return value2

def rle_decode(mask_rle, shape):
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





