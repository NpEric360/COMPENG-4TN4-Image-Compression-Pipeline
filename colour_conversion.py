
import numpy as np

def rgb_to_yuv(rgb_img):
    #Extract image dimensions
    height, width, depth = rgb_img.shape
    #Initialize output image as numpy arrays of zeroes
    yuv_img = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            # Extract RGB values from the input image
            R, G, B = rgb_img[y, x, 0], rgb_img[y, x, 1], rgb_img[y, x, 2]

            # Convert RGB to YUV 
            #***IMPORTANT: channels U,V are shifted +128 to go from [-128,127] to [0,255]; this will result in loss of color after quantization in up and downsampling
            Y = int(0.299*R + 0.587*G + 0.114*B)
            U = int(-0.14713*R - 0.28886*G + 0.436*B+128)
            V = int(0.615*R - 0.51499*G - 0.10001*B+128)

            # Set the YUV values in the output imag0.5e/1000).clip(0,255)
            yuv_img[y, x, 0] = Y
            yuv_img[y, x, 1] = U
            yuv_img[y, x, 2] = V

    return yuv_img