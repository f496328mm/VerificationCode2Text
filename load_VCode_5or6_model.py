
import os
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#-------------------  start importing keras module ---------------------
# DL model packages
import keras.utils.np_utils as kutils
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


#=====================================================================
# test
'''

model = main()


'''
#=====================================================================


def main():

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
    
    optimizer = RMSprop(lr=1e-4, rho=0.8, epsilon=1e-05, decay=0.0)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()

    os.chdir('/home/linsam/project/fb_chatbot/verification_code2text')                  
    model.load_weights('cnn_weight/VCode_5or6.h5') 
    
    return model












