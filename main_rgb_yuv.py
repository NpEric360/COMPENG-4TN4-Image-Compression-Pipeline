import numpy as np
from PIL import Image

import colour_conversion
import up_down_sampling
import display_image
import psnr_mse

#PART 0: Load original image
print("#PART 0: Load original image")
img_RGB= np.array(Image.open('test_image.png'))
array_size = img_RGB.shape #image is currently imported as RGB

#PART 1: Convert img from RGB TO YUV Color space
print("PART 1: Convert img from RGB TO YUV Color space")
img_YUV = colour_conversion.rgb_to_yuv(img_RGB)
#SPLIT UP CHANNELS
b_y = img_YUV[:,:,0]
b_u = img_YUV[:,:,1]
b_v = img_YUV[:,:,2]


#Part 4: Convert the decompressed YUV image to RGB and display image
print("Part 4: Convert the decompressed YUV image to RGB and display image")
final_image = display_image.display_converted_image(img_YUV)

#Part 5: Compute MSE, PSNR
print("Part 5: Compute MSE, PSNR")
MSE, PSNR = psnr_mse.MSE_PSNR(img_RGB,final_image)
print(MSE)
print("PSNR = ", PSNR)