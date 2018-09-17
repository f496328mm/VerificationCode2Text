# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:09:24 2018

@author: linsam
"""

import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
from VerificationCode2Text import SelectModel5or6
from VerificationCode2Text import load_VCode56_model
# my function / class
#============================================


#===============================================================

'''
import cv2
import matplotlib.pyplot as plt

os.chdir('/home/'+ path +'/github/VerificationCode2Text/test_data/')

image_name = '0ACQP9.jpg'
image = cv2.imread(image_name)
plt.imshow(image)
label_amount = SelectModel5or6.main(image)
print(label_amount)
'''

def main(image):
#def verification_code_to_text(image_name):
    #os_path = os.getcwd()
    # os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text/build_model/test_data/')
    label_amount = SelectModel5or6.main(image)
    model = load_VCode56_model.main(label_amount)
    
    return model



