
import numpy as np

#PART 2: Downsample/Decode image

#Create a function that pads the image to ensure that the height,width are 
#divisible by the downsampling factor
def pad_image(image, factor):
    height, width = image.shape[:2]
    pad_height = factor - (height % factor)
    pad_width = factor - (width % factor)
    if pad_height == factor:
        pad_height = 0
    if pad_width == factor:
        pad_width = 0
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')


def pad_image(img,kernel,downsample_factor):
    # Take input dimensions and scale outout dimensions
    input_height, input_width = img.shape[0], img.shape[1]
    
    # Compute the padding sizes; add extra padding if dimensions is not divisible by downsample factor to prevent out of bounds errors
    #If the original image height/width is NOT divisble by downsample factor, we must add an extra layer because we will end up
    #convolving the kernel over pixels outside of the image that don't exist.
    
    top_pad = kernel.shape[0] // 2
    if (input_height % downsample_factor == 0):
        bottom_pad = top_pad 
    else:
        bottom_pad = top_pad + 1
        
    left_pad = kernel.shape[1] // 2
    if (input_width % downsample_factor == 0):
        right_pad = left_pad 
    else:
        right_pad = left_pad + 1
    
    # image is padded with zeroes
    padded_image = np.zeros((input_height + top_pad + bottom_pad, input_width + left_pad + right_pad), dtype=np.float32)
    #replace the non padded section of padded_image with the original image, = put input image in middle of padded_image
    padded_image[top_pad:top_pad+input_height, left_pad:left_pad+input_width] = img
    
    return padded_image



#Create a downsampling kernel function
def create_downsampling_kernel(kernel_size):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    return kernel


def downsample_image(img, downsample_factor):
    input_height, input_width = img.shape[0], img.shape[1]
    
    # Create downsampling kernel
    kernel = create_downsampling_kernel(downsample_factor)
    
    # Create an numpy array with downscaled dimensions filled with zeros:
    output_height, output_width = int(input_height/downsample_factor), int(input_width/downsample_factor)
    downsampled_channel = np.zeros((output_height, output_width), dtype=np.uint8)
    
    #Pad the input image to prevent out of bounds issues with kernel size mismatch with input input size
    padded_image = pad_image(img,kernel,downsample_factor)

    # Iterate over each pixel of the PADDED input image
    for i in range(output_height):
        for j in range(output_width):
            
            # Extract the section of pixels that surround the currently selected pixel with the same shape of kernel
            surrounding_pixels = padded_image[i*downsample_factor:i*downsample_factor + kernel.shape[0], j*downsample_factor:j*downsample_factor + kernel.shape[1]]
            
            # Perform element-wise multiplication between the patch and the kernel
            convoluted_section = surrounding_pixels * kernel
            
            # Sum the products of the multiplication
            final_pixel = np.sum(convoluted_section, axis=(0, 1))
            
            #quantize final_pixel to (0,255), and assign to downsampled_channel
            final_pixel = int(final_pixel)
            if (final_pixel <0):
                final_pixel = 0
            elif (final_pixel >255):
                final_pixel = 255
            downsampled_channel[i,j] = final_pixel
    
    return downsampled_channel

#this function will call downsample_image() for each YUV channel.
#I could have technically combined it all into one function, but I find that this is much easier to read and debug
#Since combining it all in one function resulted in two nested for loops, 3 seperate output np arrays, 2 kernels to be created in one single function.
def downsample(img, downsample_factor):
    downsampled_y = downsample_image(img[:,:,0],downsample_factor)
    downsampled_u = downsample_image(img[:,:,1],downsample_factor*2)
    downsampled_v = downsample_image(img[:,:,2],downsample_factor*2)
    return downsampled_y,downsampled_u,downsampled_v


#bilinear interpolation:
#Nearest neighbour method
#f(x,y) = (1-wx)*(1-wy)*f(x0,y0) + wx*(1-wy)*f(x1,y0) + (1-wx)*wy*f(x0,y1) + wx*wy*f(x1,y1)

#Where:
#Inputs = x,y coordinate
#Coordinates: x0,y0,x1,y1 are the coordinates of the four nearest pixels to input (x,y)
#x0,y0 is the coordinate of the pixel left of input x,y
#(x1,y1) is the cooridnate of the pixel right of (x,y)
#Weights = wx, wy are claculated by:
#wx = x-x0
#wy = y-y0
#This means that the weights are proportional to the 1-D distance; closest neighbour = highest weight since in the original function we take (1-wx) and/or (1-wy)

#I decided to combine all 3 YUV channel conversions in one function unlike downsample because I was not running into index out of bounds issues
#probably because the kernel convolution in downsampling caused me trouble.
def upsample_yuv_image(Y, U, V, upsample_factor):
    # Get the shape of the input image
    height, width = Y.shape
    
    # Create the output image
    new_height = (height * upsample_factor)
    new_width = (width * upsample_factor)
    print(new_height,new_width)
    Y_out = np.zeros((new_height, new_width), dtype=Y.dtype)
    U_out = np.zeros((new_height, new_width), dtype=U.dtype)
    V_out = np.zeros((new_height, new_width), dtype=V.dtype)
    
    # Upsample the Y channel
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the input pixel coordinates
            x = i / upsample_factor
            y = j / upsample_factor
            
            # Calculate the four nearest input pixels
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, height - 1)
            y2 = min(y1 + 1, width - 1)
            
            # Calculate the weights for each input pixel
            w1 = (x2 - x) * (y2 - y)
            w2 = (x2 - x) * (y - y1)
            w3 = (x - x1) * (y2 - y)
            w4 = (x - x1) * (y - y1)
            
            # Upsample the pixel using bilinear interpolation
            Y_out[i, j] = w1 * Y[x1, y1] + w2 * Y[x1, y2] + w3 * Y[x2, y1] + w4 * Y[x2, y2]
    
    # Upsample the U and V channels
    for i in range(new_height):
        for j in range(new_width):
            # Calculate the input pixel coordinates
            x = i / (upsample_factor * 2)
            y = j / (upsample_factor * 2)
            
            # Calculate the four nearest input pixels

            x1 = max(min(int(np.floor(i / (upsample_factor*2))), U.shape[0] - 2), 0)
            x2 = max(min(x1 + 1, U.shape[0] - 1), 0)
            y1 = max(min(int(np.floor(j / (upsample_factor*2))), U.shape[1] - 2), 0)
            y2 = max(min(y1 + 1, U.shape[1] - 1), 0)
            
            # Calculate the weights for each input pixel
            w1 = (x2 - x) * (y2 - y)
            w2 = (x2 - x) * (y - y1)
            w3 = (x - x1) * (y2 - y)
            w4 = (x - x1) * (y - y1)
            
            #if (j%10 ==0):
            #    print(i,j,x,y,x1,y1,x2,y2)
            # Upsample the pixel using bilinear interpolation
            #print(x,y,x1,y1,x2,y2)
            U_out[i, j] = w1 * U[x1, y1] + w2 * U[x1, y2] + w3 * U[x2, y1] + w4 * U[x2, y2]
            V_out[i, j] = w1 * V[x1, y1] + w2 * V[x1, y2] + w3 * V[x2, y1] + w4 * V[x2, y2]
    
    return Y_out, U_out, V_out

