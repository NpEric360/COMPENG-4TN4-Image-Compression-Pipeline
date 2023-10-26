import numpy as np


def downsample_channel(channel, factor):
    # Apply the downsampling by averaging the pixels in the factor x factor block
    kernel = np.ones((factor, factor), dtype=np.float32) / (factor * factor)
    downsampled = np.zeros((channel.shape[0] // factor, channel.shape[1] // factor), dtype=np.uint8)
    for i in range(downsampled.shape[0]):
        for j in range(downsampled.shape[1]):
            final_pixel = np.sum(channel[i*factor:(i+1)*factor, j*factor:(j+1)*factor] * kernel)
            #final_pixel = int(final_pixel)
            if (final_pixel <0):
                final_pixel = 0
            elif (final_pixel >255):
                final_pixel = 255
            downsampled[i, j] = final_pixel
    return downsampled

def downsample_with_kernel(image, factor):
    height, width = image.shape[:2]
    new_height, new_width = int(height/factor), int(width/factor)
    kernel = np.ones((factor, factor), dtype=np.float32) / (factor * factor)
    output = np.zeros((new_height, new_width), dtype=np.float32)
    for i in range(new_height):
        for j in range(new_width):
            final_pixel = np.sum(image[i*factor:(i+1)*factor, j*factor:(j+1)*factor] * kernel, axis=(0, 1))
            if (final_pixel <0):
                final_pixel = 0
            elif (final_pixel >255):
                final_pixel = 255
            output[i, j] = final_pixel
    return output.astype(np.uint8)


def downsampler(img, scale_factor):
    # Determine the size of the output image
    out_shape = (int(img.shape[0] / scale_factor), int(img.shape[1] / scale_factor))
    
    # Create an empty output image
    out_img = np.zeros(out_shape, dtype=img.dtype)
    
    # Compute the area of each input pixel that contributes to each output pixel
    area = np.ones((scale_factor, scale_factor), dtype=np.float32)
    area /= scale_factor ** 2
    
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            x_start = i * scale_factor
            x_end = (i+1) * scale_factor - 1
            y_start = j * scale_factor
            y_end = (j+1) * scale_factor - 1
            out_img[i, j] = np.sum(img[x_start:x_end+1, y_start:y_end+1] * area)
    
    return out_img

