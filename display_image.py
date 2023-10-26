import numpy as np
from PIL import Image

#PART 5: Convert RGB TO YUV and Display decompressed image for testing

def display_converted_image(yuv_img):
    #extract dimensions of input image
    height, width, _ = yuv_img.shape
    #initialize an empty numpy array for the RGB image to be filled with converted values
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Extract Y, U, V values from the YUV image
            Y, U, V = yuv_img[y, x, 0], yuv_img[y, x, 1], yuv_img[y, x, 2]

            # Convert YUV to RGB
            R = int(Y + 1.13983*(V-128))
            G = int(Y - 0.39465*(U-128) - 0.58060*(V-128))
            B = int(Y + 2.03211*(U-128))

            # Restrict RGB values to [0, 255] range
            R = max(0, min(R, 255))
            G = max(0, min(G, 255))
            B = max(0, min(B, 255))

            # Set the RGB value in the output image
            rgb_img[y, x, 0] = R
            rgb_img[y, x, 1] = G
            rgb_img[y, x, 2] = B

    # Display the RGB image using PIL
    pil_img = Image.fromarray(rgb_img)
    pil_img.show()
    pil_img.save("final_image.png")
    return rgb_img
