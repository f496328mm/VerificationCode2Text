
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import time
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
import build_VCode_5or6_model
import ImageProcessing

#os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text') # 設定資料夾位置
#os.getcwd()
#sys.path.getwd()
#=====================================================================
#=====================================================================
'''
import time
work_vcode_fun(20000,'train_data_5',5)
work_vcode_fun(20000,'train_data_6',6)

work_vcode_fun(2000,'test_data_5',5)
work_vcode_fun(2000,'test_data_6',6)


# 0.99645 0.99475
'''
#===============================================================
'''
self = InputData('train_data_5',10,5)
self.main()
plt.imshow(self.data_set[1])
self.real_labels[1]
self.labels_set[0][1]
self.labels_set[1][1]
self.labels_set[2][1]
self.labels_set[3][1]
self.labels_set[4][1]
'''
# plt.imshow()
class InputData(build_VCode_5or6_model.InputData):
    
    def __init__(self,file_path,data_amount,label_amount):
        build_VCode_5or6_model.InputData.__init__(self,file_path,data_amount)
        
        total_set = []
        for i in range(65, 91):
            total_set.append( chr(i) )
        for i in range(10):
            total_set.append(str(i))

        self.total_set = total_set
        self.label_amount = label_amount
        
    '''def input_train_data(self):
        #-------------------------------------------------------
        sum_count = len(self.train_image_path)
        data_set = np.ndarray(( sum_count , 60, 200,3), dtype=np.uint8)
        i=0
        #s = time.time()
        while( i < sum_count ):
            image_name = self.train_image_path[i]
            image = cv2.imread(image_name)
            #plt.imshow(image)
            image = ImageProcessing.main(image)
            #plt.imshow(image)
            data_set[i] = image
            #labels_set[i] = k
            i=i+1
            if i%50 == 0: print('Processed {} of {}'.format(i, sum_count ) )
        #t = time.time() -s
        #print(t)                
        #labels_set = np.ndarray( sum_count , dtype=np.uint8)
        
        self.data_set = data_set'''
    #-------------------------------------------------------
    # label to ont hot encoder
    # Y要是一個含有6個numpy array的list
    def input_train_labels(self):
        
        def change_onehotencoder(text_set,total_set):
                
            total_amount = len(total_set)
            labellist = []
            
            for number in text_set:
                #print('number : '+str(number))
                onehot = [0 for _ in range(total_amount)]
                for i in range(len(total_set)):
                    if number == total_set[i]:
                        onehot[i] = 1
                        labellist.append(onehot)
                        break
    
            return labellist
            
        labels_set = [[] for _ in range( self.label_amount )]
        real_labels = []
        for text in self.train_image_path:
            text = text.split('/')
            text = text[len(text)-1]
            #print(text)
            #real_labels = 
            text_set = text.replace('.jpg','')
            real_labels.append(text_set)
            
            text_set = change_onehotencoder(text_set,self.total_set)
            for i in range( self.label_amount ):
                labels_set[i].append(text_set[i])
             
        self.labels_set = [arr for arr in np.asarray(labels_set)]# 最後要把6個numpy array 放在一個list
        self.real_labels = real_labels
        
    def main(self):
        self.input_train_data()
        self.input_train_labels()
        # plt.imshow(self.data_set[1])
        
'''

build_vcode5 = build_VCode_model(
                'train_data_5',
                'test_data_5',
                20000,
                2000,
                5,
                'VCode5')

build_vcode5.main()
print(build_vcode5.train_final_currect,build_vcode5.test_final_currect)
# train [0.9985, 0.9991, 0.9979, 0.9978, 0.994] 0.9874
# test [0.9555, 0.9615, 0.96, 0.966, 0.9705] 0.8275



 
build_vcode6 = build_VCode_model(
                'train_data_6',
                'test_data_6',
                40000,
                4000,
                6,
                'VCode6')

build_vcode6.main()    
print(build_vcode6.train_final_currect,build_vcode6.test_final_currect)

train [0.999, 0.9995, 0.99, 0.999, 0.9997, 0.999] 0.9977
test [0.9966, 0.996, 0.9974, 0.9982, 0.9968, 0.9972] 0.9828

                
'''

class build_VCode_model(build_VCode_5or6_model.build_VCode_5or6_model):
    
    def __init__(self,
                 train_file_path,
                 test_file_path,
                 train_amount,
                 test_amount,
                 label_amount,
                 weight_name):
                     
        build_VCode_5or6_model.build_VCode_5or6_model.__init__(self,train_amount,test_amount)
        
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        
        self.train_amount = train_amount
        self.test_amount = test_amount
        
        self.label_amount = label_amount
        self.weight_name = weight_name

    def InputTrainData(self):
        #file_path = 'train_data_5'
        input_data = InputData(self.train_file_path,self.train_amount,self.label_amount)
        input_data.main()
        
        self.total_set = input_data.total_set
        self.train_set = input_data.data_set
        self.train_labels = input_data.labels_set
        self.train_real_labels = input_data.real_labels

    def InputTestData(self):
        #file_path = 'test_data'
        input_data = InputData(self.test_file_path,self.test_amount,self.label_amount)
        input_data.main()
        
        self.test_set = input_data.data_set
        self.test_labels = input_data.labels_set
        self.test_real_labels = input_data.real_labels

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
            
            tensor_out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            
            Dense(1024, activation = "relu")            
            
            tensor_out = Flatten()(tensor_out)
            tensor_out = Dropout(0.5)(tensor_out)

            tem = []
            for i in range(self.label_amount):
                tem.append(Dense(36, name='digit'+str(i), activation='softmax')(tensor_out))

            tensor_out = tem
            model = Model(inputs=tensor_in, outputs=tensor_out)

            return model

        model = build_cnn_model()#self.label_amount
        #===============================================================
        optimizer = RMSprop(lr=1e-3, rho=0.8, epsilon=1e-08, decay=0.0)
        # Adamax
        # Define the optimizer
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.summary()

        history = model.fit(self.train_set,self.train_labels, 
                            batch_size = 512, epochs=50, verbose=1, 
                            validation_data=(self.test_set,self.test_labels) )
        
        self.model = model
        self.history = history

    def compare_val_train_error(self):
        # init        
        total_set = self.total_set
        model = self.model
        label_amount = self.label_amount
        #-----------------------------------------------
        def change_character(pred_prob,total_set):
                    
            total_amount = len(total_set)
    
            for i in range(len(pred_prob)):
                if pred_prob[i] == max( pred_prob ):
                    value = (total_set[i])
                    
            return value
        #-----------------------------------------------
        def compare_error(data,labels):
            #----------------------------------------------------------------
            # data = self.test_set
            # labels = self.test_real_labels
            prediction = model.predict(data, verbose=1)
            amount = len(labels)
            resultlist = ["" for _ in range(amount)]
            # array change to real labels
            for i in range(amount):
                for j in range(len(prediction)):
                    #print(j)
                    resultlist[i] += change_character(prediction[j][i],total_set)
                    
            #resultlist[:10]
            #----------------------------------------------------------------
            # compare case 1,2,3,4,5 error
            total = len(resultlist)
            # init score
            score = [0 for _ in range(label_amount)]
            final_score = 0
            
            for i in range(total):
                #print(i)
                if resultlist[i] == labels[i]: final_score = final_score+1
                for j in range(label_amount):
                    #print(resultlist[i][j])
                    try:
                        if resultlist[i][j] == labels[i][j]: 
                            score[j] += 1   
                    except:
                        123
            for i in range(label_amount):
                score[i] /= total
                
            final_score = final_score/total    
            #---------------------------------------------------------------
            return score,final_score
        #----------------------------------------------------------------
        # main
        self.train_currect_per,self.train_final_currect = compare_error(self.train_set,self.train_real_labels) 
        self.test_currect_per,self.test_final_currect = compare_error(self.test_set,self.test_real_labels) 
        print()
        print('train',self.train_currect_per,self.train_final_currect)
        print('test',self.test_currect_per,self.test_final_currect)
    
    def show_history_plot(self):#(train = 'acc', validation = 'val_acc'):
        plt.figure(figsize = (10,10)) # change figure size
        
        for i in range(self.label_amount,1):
            print(i)
            plt.plot( self.history.history['digit'+str(i)+'_acc'] )
            plt.plot( self.history.history['val_digit'+str(i)+'_acc'] )
            plt.title('train history')
            plt.ylabel('acc'+str(i))
            plt.xlabel('Epoch'+str(i))

                    
    def main(self):
        self.InputTrainData()
        self.InputTestData()
        
        self.train_verification_model()
        
        self.compare_val_train_error()
        self.show_history_plot()
        
        os.chdir('/home/linsam/project/fb_chatbot/verification_code2text')
        self.model.save_weights('cnn_weight/'+self.weight_name+'.h5')
        
        # 0.995475 0.992875

        
        
    