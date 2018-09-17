#==========================================================

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')

#============================function============================

# 對其一個點的周圍 2x2 的範圍進行掃描，並設定 threshold 
# 若高於 threshold 則視為非雜點
def del_mis_pt_by_threshold(im2):
    for col in range(3):
        count = 0
        for i in xrange(len(im2)):
            for j in xrange( len(im2[i]) ):
                if( im2[i,j,col] == 255):
                   count = 0 
                   for k in range(-2,3):
                       #print(k)
                       for l in range(-2,3):
                           try:
                               if im2[i + k,j + l,col] == 255:
                                   count += 1
                           except IndexError:
                               pass
        			# 這裡 threshold 設 4，當周遭小於 4 個點的話視為雜點
                if count <= 4:
                    im2[i,j,col] = 0
    return im2

def tran_gray(jpg_path):
    #jpg_path = 't1.jpg'
    image = Image.open(jpg_path)
    #灰度化处理
    #有很多种算法，这里选择rgb加权平均值算法
    gray_image = Image.new('L', image.size)
    #获得rgb数据序列，每一个都是(r,g,b)三元组
    raw_data = image.getdata()
    gray_data = []
    for rgb in raw_data:
        value = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        #value就是灰度值，这里使用127作为阀值，
        #小于127的就认为是黑色也就是0 大于等于127的就是白色，也就是255
        if value < 127:
            gray_data.append(0)
        else:
            gray_data.append(255)
    
    gray_image.putdata(gray_data)
    return gray_image

def xrange(x):

    return iter(range(x))

def input_image_pix_and_plot(jpg_path):
    #jpg_path = 't1.jpg'
    image = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
    plt.imshow(image)
    plt.show()
    return image

def catch_axis_start_and_end(im4,axis='x'):

    if(axis=='x'):
        split = []
        for i in range(im4.shape[1]):
            if( sum( im4[:,i,0] ) != 0 ):
                split.append(i)
                
        split_num_start = [split[0]]
        split_num_end = []     
           
        for i in range(2,len(split),1):
            #print(i)
            if(split[i]-split[i-1]>3): #and split[i-1]-split[i-2]!=1):
                split_num_end.append(split[i-1])
                split_num_start.append(split[i])
                #print(split[i])
        split_num_end.append(split[len(split)-1])
    elif(axis=='y'):
        split = []
        for i in range(im4.shape[0]):
            if( sum( im4[i,:,0] ) != 0 ):
                split.append(i)
        split_num_start = min(split)
        split_num_end = max(split)
    
    return split_num_start,split_num_end

#============================main============================

# input image
'''
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
from VerificationCode2Text.work_vcode import work_vcode_fun
import random

work_vcode_fun(50000,'train_data_5',5)
work_vcode_fun(50000,'train_data_6',6)

work_vcode_fun(5000,'test_data',5)
work_vcode_fun(5000,'test_data',6)



file_path = 'test_data'
file_path = ''/home/'+ path +'/github/VerificationCode2Text' + file_path + '/'
train_image_path = [file_path + i for i in os.listdir(file_path+'/')]

image_name = train_image_path[random.sample( range(10000) ,1)[0]]

image = cv2.imread(image_name)
plt.imshow(image)


new_image = main(image)
plt.imshow(new_image)




'''
def main(image):
# transform gray scale
    retval, image2 = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY_INV)
    #plt.imshow(image2)
    
    # del miscellaneous points
    image3 = del_mis_pt_by_threshold(image2)  
    #plt.imshow(image3)
    
    # image enhancement
    # 雜點去除後，我們必須對剩下的影像做強化
    image4 = cv2.dilate(image3, (2, 2), iterations=1)
    #plt.imshow(image4)
# save figure
    return image4

