import numpy as np
def RGB_to_GRAYSCALE(image):
    height, width, channels = image.shape
    # Initialize empty np array
    gray_image = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate over each pixel in the image and compute its grayscale value
    for i in range(height):
        for j in range(width):
            # Get the red, green, and blue values of the pixel
            r, g, b = image[i,j]
            # Compute the grayscale value using the luminance method
            y = 0.2126*r + 0.7152*g + 0.0722*b
            # Store the grayscale value in the new image array
            gray_image[i,j] = int(y)
    return gray_image