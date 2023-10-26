import numpy as np
from PIL import Image

import colour_conversion
import up_down_sampling
import display_image
import psnr_mse
import os
import tensorflow as tf
from tensorflow import keras

import train_CNN
import downsample
import SSIM
import rgb_to_grayscale
#PART 0: Load original image
print("#PART 0: Load original image")
#img_RGB= np.array(Image.open('test_image.png'))

img_RGB= np.array(Image.open('kodim19_512x512.png'))
#img_RGB= np.array(Image.open('dress_512x512.png'))

array_size = img_RGB.shape #image is currently imported as RGB

#PART 1: Convert img from RGB TO YUV Color space
print("PART 1: Convert img from RGB TO YUV Color space")
img_YUV = colour_conversion.rgb_to_yuv(img_RGB)
#SPLIT UP CHANNELS
b_y = img_YUV[:,:,0]
b_u = img_YUV[:,:,1]
b_v = img_YUV[:,:,2]


#PART 2: Downsample YUV image
print("PART 2: Downsample YUV image")

downsampled_y = downsample.downsampler(b_y, 2)
downsampled_u = downsample.downsampler(b_u, 4)
downsampled_v = downsample.downsampler(b_v, 4)

print(downsampled_y.shape,downsampled_u.shape,downsampled_v.shape)

#Part 3: Upsampling (Decoding)
print("Part 3: Upsampling (Decoding) with CNN")
#load trained model and call .predict() with downsampled channels

#loaded_model = tf.keras.models.load_model('trained_model_rgb_out')
loaded_model = tf.keras.models.load_model('trained_model_rgb_out_v7')

y_img,u_img,v_img = downsampled_y,downsampled_u,downsampled_v

# NEED TO match shape of tensorflow input tensor; # of numbers, height, width, # channel
y = y_img[np.newaxis,:,:,np.newaxis]/255.0
u = u_img[np.newaxis,:,:,np.newaxis]/255.0
v = v_img[np.newaxis,:,:,np.newaxis]/255.0


prediction_np = train_CNN.model_predict(loaded_model,y,u,v)
prediction = Image.fromarray(prediction_np)
prediction.show()

prediction.save(os.path.join(os.getcwd(),'final_image_phase2.png'))

#Part 4: Compute MSE, PSNR
print("Part 4: Compute MSE, PSNR + SSIM")

MSE, PSNR = psnr_mse.MSE_PSNR(img_RGB[:,:,:3],prediction_np)
print("PSNR = ", PSNR)


#Part 5: Compute SSIM

test_img_grayscale = rgb_to_grayscale.RGB_to_GRAYSCALE(img_RGB[:,:,:3])
prediction_grayscale = rgb_to_grayscale.RGB_to_GRAYSCALE(prediction_np)

SSIM_score = SSIM.SSIM(test_img_grayscale,prediction_grayscale)
print('SSIM = {}'.format(SSIM_score))
