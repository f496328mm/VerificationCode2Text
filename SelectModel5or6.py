# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:09:24 2018

@author: linsam
"""


import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
from VerificationCode2Text import load_VCode_5or6_model
# my function / class
#============================================
import numpy as np

#===============================================================


'''
import cv2
import matplotlib.pyplot as plt

tem = '/home/'+ path +'/github/VerificationCode2Text/test_data/' 
image_name = tem + '1D5GE.jpg'
image = cv2.imread(image_name)
plt.imshow(image)
Select_Model = main(image)
print(Select_Model)
'''

def main(image):
#def verification_code_to_text(image_name):
    os_path = os.getcwd()
    # os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text/build_model/test_data/')
    train_set = np.ndarray(( 1 , 60, 200,3), dtype=np.uint8)
    #image = cv2.imread(image_name)
    train_set[0] = image

    model = load_VCode_5or6_model.main()

    result = model.predict(train_set)[0]
    
    if max(result)==result[1]:
        value = 6
    elif max(result)==result[0]:
        value = 5
        
    os.chdir(os_path)
    
    return value



