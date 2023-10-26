# Image-Compression-Pipeline

This is an image compression pipeline that uses a convolutional neural network trained on images before and after being downsampled and converted into the YUV color space. 

##Part A: Prepare the dataset for training:
Step 1: Convert RGB images into YUV colour space
-> Colour_conversion.py: Extract each colour channel's values and pperform the following transformations:
            Y = int(0.299*R + 0.587*G + 0.114*B)
            U = int(-0.14713*R - 0.28886*G + 0.436*B+128)
            V = int(0.615*R - 0.51499*G - 0.10001*B+128)

Step 2: Downsample YUV Image
-> Downsample.py: Nearest neighbour averaging is used with a kernel to average the dot product of a section of the input image and the kernel. I.e. a 2x2 pixel section would become a single pixel in the resulting image.
-> Note that the U, V channels are down sampled twice as much as the Y channel since the Y channel contains more meaningful information in the image.

Step 3: Data preprocessing:
  process_data_phase1.py: Make all the necessary directories to generate the training and validation data where the individual RGB, and YUV numpy arrays will be saved.

## Part B: Construct and Train the Convolutional Neural Network
  Step 1: Construct model
  -> train_CNN_2.py: Reasoning behind the model structure on page 5 of the report.
  
## Part C: Test the model  
  Step 1: Train model on generated data and test on sample data:
  -> main.py
  The prediction accuracy is measured in Mean Squared Error (MSE) and peak signal-to-noise ratio (PSNR).
