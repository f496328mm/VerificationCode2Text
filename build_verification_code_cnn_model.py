
import cv2
import numpy as np
import matplotlib.pyplot as plt
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#-------------------  start importing keras module ---------------------
from keras.optimizers import RMSprop
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
# os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text') # 設定資料夾位置
#os.getcwd()
#=====================================================================
# test
'''
from VerificationCode2Text import work_vcode
work_vcode.work_vcode_fun(50,'train_data',5)
work_vcode.work_vcode_fun(50,'train_data',6)
work_vcode.work_vcode_fun(10,'test_data',5)
work_vcode.work_vcode_fun(10,'test_data',6)

s = time.time() 
tem = build_verification_code_cnn_model()
tem.build_model_process()

t = time.time() -s
print(t)

'''
#=====================================================================
# self = build_verification_code_cnn_model()
class build_verification_code_cnn_model:
    
    def __init__(self):
        total_set = []
        for i in range(65, 91):
            total_set.append( chr(i) )
        for i in range(10):
            total_set.append(str(i))
        total_set.append('null')
        
        self.total_set = total_set
        
    def build_model_process(self):
        self.train_data,self.train_labels = self.input_data('train_data',200000)
        self.test_data,self.test_labels = self.input_data('test_data',40000)

        self.train_verification_model()

        print( self.train_correct3,'\n' , self.test_correct3 )
        print( self.train_final_score,'\n', self.test_final_score )
        # 20000 data,  final correct : 0.88975  0.68625
        # 50000 data,  0.9784 0.8705, epochs = 40
        # 150000 data,  0.9603 0.914, epochs = 40
        # 50000 data,  0.99038 0.8793, epochs = 80 
        # 100000 data, 0.97102 0.9072, epochs=40
        # 100000 data, 0.97123 0.90295, epochs=50
        
        self.show_train_history()
        
        #os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text')
        tem = '/home/'+ path +'/github/VerificationCode2Text/'
        self.model.save_weights(tem + 'cnn_weight/verificatioin_code.h5')
                
    def load_data_and_translate_type_for_DL(self):

        self.input_data()
        
        self.train_set = self.train_set / 255.0
    #===============================================================
    
    def input_data(self,file_path,n):
        
        #os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text'+
        #'/build_model') # 設定資料夾位置
        # file_path = 'train_data_5'; n = 10
        tem = '/home/'+ path +'/github/VerificationCode2Text/'
        file_path = tem + file_path+'/'
        train_image_path = [file_path + i for i in os.listdir(file_path+'/')][:n]
        #-------------------------------------------------------
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
            if  len(labellist) == 5:
                onehot = [0 for _ in range(total_amount)]
                #print('null')
                onehot[len(total_set)-1] = 1
                labellist.append(onehot)
                        

            return labellist
        #-------------------------------------------------------
        # label to ont hot encoder
        # Y要是一個含有6個numpy array的list
        labels_set = [[] for _ in range(6)]

        for text in train_image_path:
            text = text.replace(file_path,'')
            text_set = text.replace('.jpg','')
            text_set = change_onehotencoder(text_set,self.total_set)
            for i in range(6):
                labels_set[i].append(text_set[i])
             
        labels_set = [arr for arr in np.asarray(labels_set)]# 最後要把6個numpy array 放在一個list
        #-------------------------------------------------------
        # input image to array
        sum_count = len(train_image_path)
        train_set = np.ndarray(( sum_count , 60, 200,3), dtype=np.uint8)
        #labels_set = np.ndarray( sum_count , dtype=np.uint8)
        
        i=0
        while( i < sum_count ):
            image_name = train_image_path[i]
            image = cv2.imread(image_name)
            #plt.imshow(im)
            train_set[i] = image
            #labels_set[i] = k
            i=i+1
            if i%50 == 0: print('Processed {} of {}'.format(i, sum_count ) )
    
        return train_set,labels_set

    #===============================================================
    def train_verification_model(self):

        def build_cnn_model():
            from keras.models import Model
            from keras.layers import Input, Conv2D, Dropout, Dense, Flatten, MaxPooling2D
            
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
            tensor_out = [Dense(37, name='digit1', activation='softmax')(tensor_out),\
                Dense(37, name='digit2', activation='softmax')(tensor_out),\
                Dense(37, name='digit3', activation='softmax')(tensor_out),\
                Dense(37, name='digit4', activation='softmax')(tensor_out),\
                Dense(37, name='digit5', activation='softmax')(tensor_out),\
                Dense(37, name='digit6', activation='softmax')(tensor_out)]
            model = Model(inputs=tensor_in, outputs=tensor_out)
            

            return model

        model = build_cnn_model()
        #===============================================================
        optimizer = RMSprop(lr=1e-3, rho=0.8, epsilon=1e-08, decay=0.0)
        # Adamax
        # Define the optimizer
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # model.summary()

        history = model.fit(self.train_data,self.train_labels, 
                            batch_size = 512, epochs=10, verbose=1, 
                            validation_data=(self.test_data,self.test_labels) )
        
        self.model = model
        self.history = history
        ( self.train_correct3 , self.test_correct3, 
          self.train_final_score, self.test_final_score ) = self.compare_val_train_error()
#-------------------------------------------------------------------
    
    def compare_val_train_error(self):
        #-----------------------------------------------
        def change_character(pred_prob,total_set):
                    
            #total_amount = len(total_set)
    
            for i in range(len(pred_prob)):
                if pred_prob[i] == max( pred_prob ):
                    value = (total_set[i])
            if value == 'null':
                return ''
            return value
        #-----------------------------------------------
        def compare_error(file_path,self,data):
            # file_path = 'train_data'
            # data = self.train_data
            #os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text'+
            #'/build_model')
            tem = '/home/'+ path +'/github/VerificationCode2Text/'
            file_path = tem + file_path+'/'
            train_image_path = [file_path + i for i in os.listdir(file_path+'/')][:len(data)]
            
            labels_set=[]
    
            for text in train_image_path:
                text = text.replace(file_path,'')
                text = text.replace('.jpg','')
                labels_set.append( text )
            #----------------------------------------------------------------
            #prediction = self.model.predict(self.train_data, verbose=1)
            prediction = self.model.predict(data, verbose=1)
            amount = len(labels_set)
            resultlist = ["" for _ in range(amount)]
            
            for i in range(amount):
                for j in range(len(prediction)):
                    #print(j)
                    resultlist[i] += change_character(prediction[j][i],self.total_set)
                    
            #resultlist[:10]
            #----------------------------------------------------------------
            total = len(resultlist)
            score = [0 for _ in range(6)]
            for i in range(total):
                #print(i)
                for j in range(6):
                    #print(resultlist[i][j])
                    try:
                        if resultlist[i][j] == labels_set[i][j]: 
                            score[j] += 1   
                    except:
                        123
            for i in range(6):
                score[i] /= total
                
            #current_per = score/total
            return score
            
        #----------------------------------------------------------------
        #-----------------------------------------------
        def compare_final_error(file_path,self,data):
            # file_path = 'train_data'
            # data = self.train_data
            tem = '/home/'+ path +'/github/VerificationCode2Text/'
            file_path = tem + file_path+'/'
            train_image_path = [file_path + i for i in os.listdir(file_path+'/')][:len(data)]
            
            labels_set=[]
    
            for text in train_image_path:
                text = text.replace(file_path,'')
                text = text.replace('.jpg','')
                labels_set.append( text )
            #----------------------------------------------------------------
            #prediction = self.model.predict(self.train_data, verbose=1)
            #prediction = self.model.predict(data, verbose=1)
            amount = len(labels_set)
            resultlist = ["" for _ in range(amount)]
            
            #for i in range(amount):
                #for j in range(len(prediction)):
                    #print(j)
            #----------------------------------------------------------------
            total = len(resultlist)
            score = 0
            for i in range(total):
                if resultlist[i] == labels_set[i]: score = score+1
            score = score/total

            return score
        #----------------------------------------------------------------
        v1 = compare_error('train_data',self,self.train_data) 
        v2 = compare_error('test_data',self,self.test_data) 
        v3 = compare_final_error('train_data',self,self.train_data)
        v4 = compare_final_error('test_data',self,self.test_data)
        
        return v1,v2,v3,v4  
    #===============================================================
    def show_train_history(self):#(train = 'acc', validation = 'val_acc'):
        plt.figure(figsize = (10,10)) # change figure size
        for i in range(1,6,1):
            #print(i)
            plt.plot( self.history.history['digit'+str(i)+'_acc'] )
            plt.plot( self.history.history['val_digit'+str(i)+'_acc'] )
            plt.title('train history')
            plt.ylabel('acc'+str(i))
            plt.xlabel('Epoch'+str(i))

        
        plt.legend(['digit1_acc','val_digit1_acc',
                    'digit2_acc','val_digit2_acc',
                    'digit3_acc','val_digit3_acc',
                    'digit4_acc','val_digit4_acc',
                    'digit5_acc','val_digit5_acc'],loc = 'upper left')
    #===============================================================
    #===============================================================




