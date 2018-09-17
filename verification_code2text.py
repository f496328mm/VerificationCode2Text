
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
from VerificationCode2Text import SelectModel56
# my function / class
#============================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
#===============================================================

def test():
    import random
    
    file_path = 'test_data'
    file_path = '/home/'+ path +'/github' + '/VerificationCode2Text/'+file_path+'/'
    train_image_path = [file_path + i for i in os.listdir(file_path+'/')]
    
    image_name = train_image_path[random.sample( range(10) ,1)[0]]
    
    image = cv2.imread(image_name)
    plt.imshow(image)
    
    text = main(image)
    print(text)



def validation(test_path):

    file_path = 'success_vcode'
    file_path = '/home/'+ path +'/github/VerificationCode2Text/'+file_path+'/'
    test_image_path = [file_path + i for i in os.listdir(file_path+'/')]
    
    sum_count = len(test_image_path)
    data_set = np.ndarray(( sum_count , 60, 200,3), dtype=np.uint8)
    i=0
    #s = time.time()
    while( i < sum_count ):
        image_name = test_image_path[i]
        image = cv2.imread(image_name)
        #plt.imshow(image)
        #image = ImageProcessing.main(image)
        #plt.imshow(image)
        data_set[i] = image
        #labels_set[i] = k
        i=i+1
        if i%50 == 0: print('Processed {} of {}'.format(i, sum_count ) )
            
#--------------------------------------------------
    real_labels = []
    for text in test_image_path:
        text = text.split('/')
        text = text[len(text)-1]
        text_set = text.replace('.png','')
        real_labels.append(text_set)

    #self.real_labels = real_labels            
    
    image = cv2.imread(image_name)
    plt.imshow(image)
    
    text = main(image)
    print(text)

def main(image):
#def verification_code_to_text(image_name):
    
    os_path = os.getcwd()
    def change_character(pred_prob):
        
        total_set = []
        for i in range(65, 91):
            total_set.append( chr(i) )
        for i in range(10):
            total_set.append(str(i))
    
        for i in range(len(pred_prob)):
            if pred_prob[i] == max( pred_prob ):
                value = (total_set[i])

        return value
    
    train_set = np.ndarray(( 1 , 60, 200,3), dtype=np.uint8)
    #image = cv2.imread(image_name)
    train_set[0] = image

    model = SelectModel56.main(image)
    result = model.predict(train_set)

    resultlist = ''
    for i in range(len(result)):
        resultlist += change_character(result[i][0])

    os.chdir(os_path)
    return resultlist



