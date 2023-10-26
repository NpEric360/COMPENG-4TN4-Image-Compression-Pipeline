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

#PART 2: Downsample 
print("PART 2: Downsample YUV image")
downsampled_y,downsampled_u,downsampled_v= up_down_sampling.downsample(img_RGB,downsample_factor = 2)

print(downsampled_y.shape,downsampled_u.shape,downsampled_v.shape)

#Part 3: Upsampling (Decoding)
print("Part 3: Upsampling (Decoding)")
y,u,v = up_down_sampling.upsample_yuv_image(downsampled_y,downsampled_u,downsampled_v,2)


#Create an empty numpy array to store the decompressed YUV image with shape (y.shape,3) 
decoded_shape = y.shape
decoded_YUV = np.zeros((decoded_shape[0],decoded_shape[1],3),dtype=np.uint8)
decoded_YUV[:,:,0]=y
decoded_YUV[:,:,1]=u
decoded_YUV[:,:,2]=v
#Part 4: Convert the decompressed YUV image to RGB and display image
print("Part 4: Convert the decompressed YUV image to RGB and display image")

pil_img = Image.fromarray(decoded_YUV)
pil_img.show()
pil_img.save("final_image.png")

#Part 5: Compute MSE, PSNR
print("Part 5: Compute MSE, PSNR")
MSE, PSNR = psnr_mse.MSE_PSNR(img_RGB,pil_img)
print(MSE)
print("PSNR = ", PSNR)