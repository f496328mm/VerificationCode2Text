
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#-------------------  start importing keras module ---------------------
# DL model packages
from keras.optimizers import RMSprop
import os
import sys
path = os.listdir('/home')[0]
sys.path.append('/home/'+ path +'/github')

#=====================================================================
# test
'''
from VerificationCode2Text import SelectModel5or6

tem = '/home/'+ path +'/github/VerificationCode2Text/test_data/'
image_name = tem + '0ACQP9.jpg'
image = cv2.imread(image_name)
plt.imshow(image)
label_amount = SelectModel5or6.main(image)
print(label_amount)

model = main(label_amount)

'''
#=====================================================================


def main(label_amount):

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
    for i in range(label_amount):
        tem.append(Dense(36, name='digit'+str(i), activation='softmax')(tensor_out))

    tensor_out = tem
    model = Model(inputs=tensor_in, outputs=tensor_out)

    #===============================================================
    optimizer = RMSprop(lr=1e-3, rho=0.8, epsilon=1e-08, decay=0.0)
    # Adamax
    # Define the optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()
 
    tem = '/home/'+ path +'/github/VerificationCode2Text/'              
    model.load_weights(tem + 'cnn_weight/VCode'+str(label_amount)+'.h5') 
    
    return model












