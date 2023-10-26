""""
train_CNN.py
This script is used to train a CNN model to receieve Y, U, V channels where the dimensions of channels U, V are half of channel Y and returns an RGB image
with a shape twice the size of channel Y.
This script will also load the model and make a prediction of a user inputted image

Directory Structure:

os.getcwd()//data
        |--DIV2K_train_HR
            |--resized
                |--y_channel (contains downsampled y channel images)
                |--u_channel (contains downsampled u channel images)
                |--v_channel (contains downsampled v channel images)

Functions:
    -make_dir(): Setup directories
    -process_data(height,width): Resize the original dataset to be uniform
    -generate_data_pairs(input,img_path): Use cv2 to convert each image to YUV and split into corresponding Y,U,V channel.
"""

import os

import cv2
from skimage.metrics import structural_similarity


import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping

import psnr_mse
from matplotlib import pyplot
import downsample
#load data

#YOU MUST /255.0 OR NN WILL NOT CONVERGE! 

y_train = np.load(os.path.join(os.getcwd(),'data_arrays','y_train.npy'))/255.0
u_train = np.load(os.path.join(os.getcwd(),'data_arrays','u_train.npy'))/255.0
v_train = np.load(os.path.join(os.getcwd(),'data_arrays','v_train.npy'))/255.0
rgb_train = np.load(os.path.join(os.getcwd(),'data_arrays','rgb_train.npy'))/255.0

rgb_train = rgb_train[:,:,:,:]

y_valid = np.load(os.path.join(os.getcwd(),'data_arrays','y_valid.npy'))/255.0
u_valid = np.load(os.path.join(os.getcwd(),'data_arrays','u_valid.npy'))/255.0
v_valid = np.load(os.path.join(os.getcwd(),'data_arrays','v_valid.npy'))/255.0
rgb_valid = np.load(os.path.join(os.getcwd(),'data_arrays','rgb_valid.npy'))/255.0


# print(y_train.shape)
# print(u_train.shape)
# print(v_train.shape)
# print(rgb_train.shape)

# print(y_valid.shape)
# print(u_valid.shape)        
# print(v_valid.shape)
# print(rgb_valid.shape)



from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, UpSampling2D, Dropout
from keras.models import Model

def construct_model():
    # Define the input shape for the model
    y_input_shape = (256, 256, 1)
    uv_input_shape = (128, 128, 1)
    # Define the input layers for the model
    y_input = Input(shape=y_input_shape, name='y_input')
    u_input = Input(shape=uv_input_shape, name='u_input')
    v_input = Input(shape=uv_input_shape, name='v_input')
    # First block of the model: convolve Y channel with 32 filters of size 3x3
    y_conv = Conv2D(128, (3, 3), padding='same')(y_input)
    y_conv = BatchNormalization()(y_conv)
    y_conv = Activation('relu')(y_conv)
    # Second block of the model: concatenate U and V channels and convolve with 32 filters of size 3x3
    uv_concat = Concatenate()([u_input, v_input])
    uv_conv = Conv2D(128, (3, 3), padding='same')(uv_concat)
    uv_conv = BatchNormalization()(uv_conv)
    uv_conv = Activation('relu')(uv_conv)
    # Third block of the model: upsample U and V channels and concatenate with Y channel
    uv_upsample = UpSampling2D(size=(2,2))(uv_conv)
    merged = Concatenate()([y_conv, uv_upsample])
    # Fourth block of the model: convolve merged channels with 64 filters of size 3x3
    conv = Conv2D(64, (3, 3), padding='same')(merged)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(0.3)(conv)
    # Fifth block of the model: upsample and convolve with 64 filters of size 3x3
    upsample = UpSampling2D(size=(2,2))(conv)
    conv = Conv2D(64, (3, 3), padding='same')(upsample)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Dropout(0.3)(conv)
    # Sixth block of the model: apply a final convolutional layer with 3 filters of size 3x3
    #sigmoid activation is for binary 0/1 classsifciation, linear is any real number
    output = Conv2D(3, (3, 3), padding='same', activation='linear')(conv)
    # Define the model with the input and output layers
    model = Model(inputs=[y_input, u_input, v_input], outputs=output)
    # Compile the model
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(optimizer=optimizer, loss='MSE')

    return model


def grab_process_test_data():

    test_img_path = os.path.join(os.getcwd(),'dress_512x512.png')
    test_img = cv2.imread(test_img_path)
    yuv_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv_img)
     
    downsample_ratio_y = 2
    downsample_ratio_uv = downsample_ratio_y*2


    downsampled_y = downsample.downsampler(y,2)
    downsampled_u = downsample.downsampler(u,4)
    downsampled_v = downsample.downsampler(v,4)
    #downsampled_y = cv2.resize(y, (y.shape[1] // downsample_ratio_y, y.shape[0] // downsample_ratio_y), interpolation=cv2.INTER_AREA)
    #downsampled_u = cv2.resize(u, (u.shape[1] // downsample_ratio_uv, u.shape[0] // downsample_ratio_uv), interpolation=cv2.INTER_AREA)
    #downsampled_v = cv2.resize(v, (v.shape[1] // downsample_ratio_uv, v.shape[0] // downsample_ratio_uv), interpolation=cv2.INTER_AREA)

    print(type(downsampled_u))

    y = downsampled_y[np.newaxis,:,:,np.newaxis]/255.0
    u = downsampled_u[np.newaxis,:,:,np.newaxis]/255.0
    v = downsampled_v[np.newaxis,:,:,np.newaxis]/255.0

    print(y.shape)
    print(u.shape)
    print(v.shape)

    #print(y)
    return y,u,v, test_img

def model_predict(model,y_img,u_img,v_img):
    output = model.predict([y_img,u_img,v_img],batch_size = 1)*255.0
    print(output.shape)
    prediction = np.uint8(output[0,:,:,:]) #1st element represents batch size
    return prediction





if __name__ == "__main__":
    model = construct_model()
    print(model.summary())

    #allow early stopping of model.fit to prevent overfitting
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit([y_train,u_train,v_train],rgb_train,
              batch_size=4,
              epochs=100, 
              callbacks = [early_stopping_callback, tf.keras.callbacks.History()],
              validation_data = ([y_valid,u_valid,v_valid],rgb_valid))
    np.save('model_history.npy',history.history)
    model.save('trained_model_rgb_out_v8')

    loaded_model = tf.keras.models.load_model('trained_model_rgb_out_v8')
    y_img,u_img,v_img,test_img= grab_process_test_data()
    prediction_np = model_predict(loaded_model,y_img,u_img,v_img)

    print(type(prediction_np))

    mse,psnr = psnr_mse.MSE_PSNR(test_img,prediction_np)
    print(psnr)

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.savefig('traing_graph')
    print('Done')