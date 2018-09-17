
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#-------------------  start importing keras module ---------------------
# test
'''

model = load_handwritten_model()


'''
#=====================================================================


def load_handwritten_model():

    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    
    tensor_in = Input((60, 200, 3))
    out = tensor_in
    out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Flatten()(out)
    out = Dropout(0.5)(out)
    out = [Dense(37, name='digit1', activation='softmax')(out),\
        Dense(37, name='digit2', activation='softmax')(out),\
        Dense(37, name='digit3', activation='softmax')(out),\
        Dense(37, name='digit4', activation='softmax')(out),\
        Dense(37, name='digit5', activation='softmax')(out),\
        Dense(37, name='digit6', activation='softmax')(out)]
    
    model = Model(inputs=tensor_in, outputs=out)
    
    # Define the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    tem = '/home/'+ path +'/github/VerificationCode2Text/'
    #os.chdir('/home/linsam/project/re_AI_order_ticket/verification_code_to_text')                  
    model.load_weights(tem + 'cnn_weight/verificatioin_code.h5') 
    
    return model












