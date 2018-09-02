
import cv2

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#-------------------  start importing keras module ---------------------
import keras.utils.np_utils as kutils
from keras.optimizers import RMSprop
sys.path.append('/home/linsam/project/fb_chatbot/verification_code2text')
from work_vcode import *
#import ImageProcessing

#os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text') # 設定資料夾位置
#os.getcwd()
#sys.path.getwd()
#=====================================================================
# test
'''
import time
work_vcode_fun(10000,'train_data_5',5)
work_vcode_fun(10000,'train_data_6',6)

work_vcode_fun(2000,'test_data',5)
work_vcode_fun(2000,'test_data',6)

s = time.time() 
tem = build_VCode_5or6_model(10000,2000)
tem.main()

t = time.time() -s
print(t/60)

# 0.99645 0.99475

'''

#=====================================================================
#===============================================================
# self = InputData('train_data_5',50)
class InputData:
    
    def __init__(self,file_path,n):
        #sys.path.append('/home/linsam/project/re_AI_order_ticket/verification_code_to_text'+
        #'/build_model/') # 設定資料夾位置
        # file_path = 'train_data_5'
        file_path = '/home/linsam/project/fb_chatbot/verification_code2text/'+file_path+'/'
        self.train_image_path = [file_path + i for i in os.listdir(file_path+'/')][:n]
    
    def input_train_data(self):
        #-------------------------------------------------------
        sum_count = len(self.train_image_path)
        data_set = np.ndarray(( sum_count , 60, 200,3), dtype=np.uint8)
        i=0
        #s = time.time()
        while( i < sum_count ):
            image_name = self.train_image_path[i]
            image = cv2.imread(image_name)
            #plt.imshow(image)
            #image = ImageProcessing.main(image)
            #plt.imshow(image)
            data_set[i] = image
            #labels_set[i] = k
            i=i+1
            if i%50 == 0: print('Processed {} of {}'.format(i, sum_count ) )
        #t = time.time() -s
        #print(t)                
        #labels_set = np.ndarray( sum_count , dtype=np.uint8)
        
        self.data_set = data_set
        
    # file_path = 'train_data_5'
    def input_train_labels(self):
        #-------------------------------------------------------
        #-------------------------------------------------------
        # label to ont hot encoder
        # Y要是一個含有6個numpy array的list
        labels_set = []
    
        for text in self.train_image_path:
            text = text.split('/')
            text = text[len(text)-1]
            
            text_set = text.replace('.jpg','')
            #print(text_set)
            if len(text_set) == 5:
                labels_set.append(0)
            elif len(text_set) == 6:
                labels_set.append(1)
                
        self.labels_set = labels_set
    
    def main(self):
        self.input_train_data()
        self.input_train_labels()
        

class build_VCode_5or6_model:
    
    def __init__(self,train_amount,test_amount):

        self.train_amount = train_amount
        self.test_amount = test_amount
        
    def InputTrainData(self,amount):
        file_path = 'train_data_5'
        input_data = InputData(file_path,amount)
        input_data.main()
        
        train_set5 = input_data.data_set
        train_labels5 = input_data.labels_set
        #--------------------------------------
        file_path = 'train_data_6'
        input_data = InputData(file_path,amount)
        input_data.main()
        
        train_set6 = input_data.data_set
        train_labels6 = input_data.labels_set
        #-------------------------------------------
        train_set = np.concatenate((train_set5,train_set6))
        train_labels5.extend(train_labels6) 
        train_labels = train_labels5
        
        train_labels = np.array(train_labels, dtype=np.uint8)
    	# one hot encoding, 0~9 轉成 01 矩陣 
        train_labels = kutils.to_categorical(train_labels) 
    
        num = random.sample(range(len(train_labels)),len(train_labels))
        
        train_set = train_set[num]
        train_labels = train_labels[num]
        
        return train_set,train_labels
        
    def InputTestData(self,amount):
        file_path = 'test_data'
        input_data = InputData(file_path,amount)
        input_data.main()
        
        test_set = input_data.data_set
        test_labels = input_data.labels_set
        #-------------------------------------------
        test_labels = np.array(test_labels, dtype=np.uint8)
    	# one hot encoding, 0~9 轉成 01 矩陣 
        test_labels = kutils.to_categorical(test_labels) 
    
        num = random.sample(range(len(test_labels)),len(test_labels))
        
        test_set = test_set[num]
        test_labels = test_labels[num]
        
        return test_set,test_labels
        
    def train_verification_model(self):
        
        def build_cnn_model():
            from keras.models import Model
            from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
            
            tensor_in = Input((60, 200, 3))
            tensor_out = tensor_in
            tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Dropout(0.25)(tensor_out)
            
            tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Dropout(0.25)(tensor_out)
            
            tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Dropout(0.25)(tensor_out)        
            
            tensor_out = Flatten()(tensor_out)
            tensor_out = Dropout(0.5)(tensor_out)
            tensor_out = Dense(2, name='digit', activation='softmax')(tensor_out)
            model = Model(inputs=tensor_in, outputs=tensor_out)
            

            return model   
        #===============================================================  
        model = build_cnn_model()

        
        optimizer = RMSprop(lr=1e-4, rho=0.8, epsilon=1e-05, decay=0.0)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.summary()

        history = model.fit(self.train_set,self.train_labels, 
                            batch_size = 512, epochs=30, verbose=1, 
                            validation_data=(self.test_set,self.test_labels) )
        
        self.model = model
        self.history = history
        
        ( self.train_correct , self.test_correct ) = self.compare_val_train_error()
        print(self.train_correct, self.test_correct)
        

    def compare_val_train_error(self):
        #-----------------------------------------------
        def change_character(pred):
                    
            if max( pred ) == pred[0] :
                    return 5
            return 6
        #-----------------------------------------------
        def compare_error(model,data,labels):
            # file_path = 'train_data'
            # data = self.train_data
            #----------------------------------------------------------------
            #prediction = self.model.predict(self.train_data, verbose=1)
            prediction = model.predict(data, verbose=1)
            amount = len(labels)
            resultlist = ["" for _ in range(amount)]
            
            for i in range(amount):
                    #print(j)
                    resultlist[i] = change_character(prediction[i])
                    
            #----------------------------------------------------------------
            #labels = self.train_labels
            total = len(resultlist)
            score=0
            for i in range(total):
                if resultlist[i] == 5 and labels[i][0]==1:
                    score = score+1
                elif resultlist[i] == 6 and labels[i][1]==1:
                    score = score+1
                    
            current_per = score/total
            
            return current_per
        #------------------------------------
        train_current = compare_error(self.model,self.train_set,self.train_labels)
        test_current = compare_error(self.model,self.test_set,self.test_labels)
        
        return train_current,test_current
            
    def show_history_plot(self):#(train = 'acc', validation = 'val_acc'):
        plt.figure(figsize = (10,10)) # change figure size
        
        plt.plot( self.history.history['acc'] )
        plt.plot( self.history.history['val_acc'] )
        plt.title('train history')
        plt.ylabel('acc')
        plt.xlabel('Epoch')

        plt.legend(['acc','val_acc'],loc = 'upper left')

    def main(self):
        self.train_set,self.train_labels = self.InputTrainData(self.train_amount)
        self.test_set,self.test_labels = self.InputTestData(self.test_amount)
        self.train_verification_model()
        self.show_history_plot()
        
        os.chdir('/home/linsam/project/fb_chatbot/verification_code2text')
        self.model.save_weights('cnn_weight/VCode_5or6.h5')
        # 0.995475 0.992875

        
        
    