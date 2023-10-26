""""
Process_data.py
This script is used to create the necessary directories and to process the DIV2K dataset in order to train the CNN in train_CNN.py

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
import numpy as np
from PIL import Image

import colour_conversion
import downsample

#NOTE FOR TA: I AM USING CV2 SIMPLY FOR DATASET PREPROCESSING, NO RESIZING OR COLOUR CONVERSION
#NUMPY AND PILLOW OFR DATA PREPROCESSING IS WAY TOO SLOW


#make directories
train_data_path = os.path.join(os.getcwd(),'data','DIV2K_train_HR','resized')
valid_data_path = os.path.join(os.getcwd(),'data','DIV2K_valid_HR','resized')

def make_dir():
    try:
        #training data
        os.makedirs(os.path.join(train_data_path,'y_channel','downsampled'),0o666)
        os.makedirs(os.path.join(train_data_path,'u_channel','downsampled'),0o666)
        os.makedirs(os.path.join(train_data_path,'v_channel','downsampled'),0o666)
        #validation data
        os.makedirs(os.path.join(valid_data_path,'y_channel','downsampled'),0o666)
        os.makedirs(os.path.join(valid_data_path,'u_channel','downsampled'),0o666)
        os.makedirs(os.path.join(valid_data_path,'v_channel','downsampled'),0o666)
        os.mkdir(os.path.join(os.getcwd(),'data_arrays'))
        print('1. Directories created.')
    except Exception as e:
        print(e)

#resize the images

def process_data(height,width):
    original_train_data_path = os.path.join(os.getcwd(),'data','DIV2K_train_HR')
    original_valid_data_path = os.path.join(os.getcwd(),'data','DIV2K_valid_HR')

    for images in os.listdir(original_train_data_path):
        if (images.endswith('.png')):
            img = cv2.imread(os.path.join(original_train_data_path,images))
            img_resized = cv2.resize(img,(height,width))
            cv2.imwrite(os.path.join(original_train_data_path,'resized',images),img_resized)

    for images in os.listdir(original_valid_data_path):
        if (images.endswith('.png')):
            img = cv2.imread(os.path.join(original_valid_data_path,images))
            img_resized = cv2.resize(img,(height,width))
            cv2.imwrite(os.path.join(original_valid_data_path,'resized',images),img_resized)
    print('2. All images have been resized to a uniform shape of {}x{}'.format(width,height))

def generate_data_pairs(input_img,path):
    downsample_ratio_y = 2
    downsample_ratio_uv = downsample_ratio_y*2
    train_data_path = path
    img_path = os.path.join(train_data_path,input_img)

    rgb_img = np.array(Image.open(img_path))
    yuv_img = colour_conversion.rgb_to_yuv(rgb_img)

    y = yuv_img[:,:,0]
    u = yuv_img[:,:,1]
    v = yuv_img[:,:,2]

    downsampled_y = downsample.downsampler(y,2)
    downsampled_u = downsample.downsampler(u,4)
    downsampled_v = downsample.downsampler(v,4)


    return downsampled_y, downsampled_u, downsampled_v
# Convert YUV image back to RGB image

original_train_data_path = os.path.join(os.getcwd(),'data','DIV2K_train_HR')
original_valid_data_path = os.path.join(os.getcwd(),'data','DIV2K_valid_HR')
#GENERATE TRAINIGN DATA

def generate_train_data():
    for images in os.listdir(original_train_data_path):
        if (images.endswith('png')):
            
            path = train_data_path
            d_y,d_u,d_v = generate_data_pairs(images,path)
            #save downsampled y channel images
            cv2.imwrite(os.path.join(path,'y_channel','downsampled',images[:-4]+'_dy'+'.png'),d_y)
            #save downsampled u channel images
            cv2.imwrite(os.path.join(path,'u_channel','downsampled',images[:-4]+'_du'+'.png'),d_u)
            #save downsampled v channel images
            cv2.imwrite(os.path.join(path,'v_channel','downsampled',images[:-4]+'_dv'+'.png'),d_v)

    print('3. Downsampled training YUV images have been generated.')
    #GENERATE VALIDATION DATA
def generate_valid_data():
    for images in os.listdir(original_valid_data_path):
        if (images.endswith('png')):
            path = valid_data_path
            d_y, d_u,d_v = generate_data_pairs(images,path)
            #save y, and downsampled y channel images
            cv2.imwrite(os.path.join(path,'y_channel','downsampled',images[:-4]+'_dy'+'.png'),d_y)
            #save u, and downsampled u channel images
            cv2.imwrite(os.path.join(path,'u_channel','downsampled',images[:-4]+'_du'+'.png'),d_u)
            #save v, and downsampled v channel images
            cv2.imwrite(os.path.join(path,'v_channel','downsampled',images[:-4]+'_dv'+'.png'),d_v)
    print('4. Downsampled validation YUV images have been generated.')
#Create numpy arrays that contain each channel's dataset to feed into NN

### Y CHANNEL
def generate_train_np():
    y_down_train = (os.path.join(os.getcwd(),'data','DIV2K_train_HR','resized','y_channel','downsampled'))
    #create a list containing all filenames
    image_filenames = [filename for filename in os.listdir(y_down_train) if filename.endswith(".png")] 

    down_y_img = Image.open(os.path.join(y_down_train,image_filenames[0]))
    d_width,d_height = down_y_img.size
    #initialize empty np array
    y_down_train_np = np.empty((len(image_filenames),d_height,d_width))

    for i,image_filename in enumerate(image_filenames):
        img = Image.open(os.path.join(y_down_train,image_filename))
        y_down_train_np[i] = np.array(img)

    ### U CHANNEL

    u_down_train = (os.path.join(os.getcwd(),'data','DIV2K_train_HR','resized','u_channel','downsampled'))
    image_filenames = [filename for filename in os.listdir(u_down_train) if filename.endswith(".png")]

    down_u_img = Image.open(os.path.join(u_down_train,image_filenames[0]))
    uv_d_width,uv_d_height = down_u_img.size

    #empty np array
    u_down_train_np = np.empty((len(image_filenames),uv_d_height,uv_d_width))

    for i,image_filename in enumerate(image_filenames):
        img = Image.open(os.path.join(u_down_train,image_filename))
        #print(img.size)
        u_down_train_np[i] = np.array(img)
    #open an image to determine height, width, channels

    ### V channel

    v_down_train = (os.path.join(os.getcwd(),'data','DIV2K_train_HR','resized','v_channel','downsampled'))
    image_filenames = [filename for filename in os.listdir(v_down_train) if filename.endswith(".png")]

    down_v_img = Image.open(os.path.join(v_down_train,image_filenames[0]))
    uv_d_width,uv_d_height = down_v_img.size
    #empty np array
    v_down_train_np = np.empty((len(image_filenames),uv_d_height,uv_d_width))

    for i,image_filename in enumerate(image_filenames):
        img = Image.open(os.path.join(v_down_train,image_filename))
        #print(img.size)
        v_down_train_np[i] = np.array(img)
    

    ### Add extra channel dimension to match model tensor shape
    y = y_down_train_np[:,:,:,np.newaxis]
    u = u_down_train_np[:,:,:,np.newaxis]
    v = v_down_train_np[:,:,:,np.newaxis]

    print('5. Training data has successfully been stored as np arrays')

    return y,u,v

##Generate Numpy array for Validation dataset

def generate_validation_np():

    y_down_valid = (os.path.join(os.getcwd(),'data','DIV2K_valid_HR','resized','y_channel','downsampled'))
    #create a list containing all filenames
    image_filenames = [filename for filename in os.listdir(y_down_valid) if filename.endswith(".png")] 

    down_y_img = Image.open(os.path.join(y_down_valid,image_filenames[0]))
    d_width,d_height = down_y_img.size
    #initialize empty np array
    y_down_valid_np = np.empty((len(image_filenames),d_height,d_width))

    for i,image_filename in enumerate(image_filenames):
        img = Image.open(os.path.join(y_down_valid,image_filename))
        y_down_valid_np[i] = np.array(img)

    ### U CHANNEL

    u_down_valid = (os.path.join(os.getcwd(),'data','DIV2K_valid_HR','resized','u_channel','downsampled'))
    image_filenames = [filename for filename in os.listdir(u_down_valid) if filename.endswith(".png")]

    down_u_img = Image.open(os.path.join(u_down_valid,image_filenames[0]))
    uv_d_width,uv_d_height = down_u_img.size

    #empty np array
    u_down_valid_np = np.empty((len(image_filenames),uv_d_height,uv_d_width))

    for i,image_filename in enumerate(image_filenames):
        img = Image.open(os.path.join(u_down_valid,image_filename))
        #print(img.size)
        u_down_valid_np[i] = np.array(img)
    #open an image to determine height, width, channels

    ### V channel

    v_down_valid = (os.path.join(os.getcwd(),'data','DIV2K_valid_HR','resized','v_channel','downsampled'))
    image_filenames = [filename for filename in os.listdir(v_down_valid) if filename.endswith(".png")]
    #y_original_train = (os.path.join(os.getcwd(),'data','DIV2K_train_HR','resized','y_channel','original'))
    down_v_img = Image.open(os.path.join(v_down_valid,image_filenames[0]))
    uv_d_width,uv_d_height = down_v_img.size
    #empty np array
    v_down_valid_np = np.empty((len(image_filenames),uv_d_height,uv_d_width))

    for i,image_filename in enumerate(image_filenames):
        img = Image.open(os.path.join(v_down_valid,image_filename))
        #print(img.size)
        v_down_valid_np[i] = np.array(img)
    
    ### Add extra channel dimension to match model tensor shape

    y_valid = y_down_valid_np [:,:,:,np.newaxis]
    u_valid = u_down_valid_np[:,:,:,np.newaxis]
    v_valid = v_down_valid_np[:,:,:,np.newaxis]

    print('6. Validation data has successfully been stored as np arrays')
    return y_valid,u_valid,v_valid

def generate_RGB_train_np():
    ### Generate numpy array for RGB pictures (label data/ground truth)
    y_original_train = (os.path.join(os.getcwd(),'data','DIV2K_train_HR','resized'))
    orig_image_filenames = [filename for filename in os.listdir(y_original_train) if filename.endswith(".png")]

    img = Image.open(os.path.join(y_original_train,orig_image_filenames[0]))
    orig_width, orig_height = img.size
    #empty np array
    rgb_orig_train_np = np.empty((len(orig_image_filenames),orig_height,orig_width,3), dtype ='uint8')

    for i,image_filename in enumerate(orig_image_filenames):
        img = Image.open(os.path.join(y_original_train,image_filename))
        rgb_orig_train_np[i] = np.array(img)
    
    print('7. Training RGB images successfully stored as np arrays')
    return rgb_orig_train_np


def generate_RGB_valid_np():
    ### Generate numpy array for RGB pictures (label data/ground truth)
    original_valid = (os.path.join(os.getcwd(),'data','DIV2K_valid_HR','resized'))
    orig_image_filenames = [filename for filename in os.listdir(original_valid) if filename.endswith(".png")]

    img = Image.open(os.path.join(original_valid,orig_image_filenames[0]))
    orig_width, orig_height = img.size
    #empty np array
    rgb_orig_valid_np = np.empty((len(orig_image_filenames),orig_height,orig_width,3), dtype ='uint8')

    for i,image_filename in enumerate(orig_image_filenames):
        img = Image.open(os.path.join(original_valid,image_filename))
        rgb_orig_valid_np[i] = np.array(img)
    
    print('8. Validation RGB images successfully stored as np arrays')
    return rgb_orig_valid_np


### Save numpy arrays to main directory to feed to NN

def save_data_as_numpy_arrays(y,u,v,y_valid,u_valid,v_valid,rgb_train,rgb_valid):
    y_train = y
    u_train = u
    v_train = v

    np.save(os.path.join(os.getcwd(),'data_arrays','y_train'),y_train)
    np.save(os.path.join(os.getcwd(),'data_arrays','u_train'),u_train)
    np.save(os.path.join(os.getcwd(),'data_arrays','v_train'),v_train)

    np.save(os.path.join(os.getcwd(),'data_arrays','y_valid'),y_valid)
    np.save(os.path.join(os.getcwd(),'data_arrays','u_valid'),u_valid)
    np.save(os.path.join(os.getcwd(),'data_arrays','v_valid'),v_valid)
    np.save(os.path.join(os.getcwd(),'data_arrays','rgb_train'),rgb_train)
    np.save(os.path.join(os.getcwd(),'data_arrays','rgb_valid'),rgb_valid)

    print('9. All numpy arrays saved to main working directory.')

if __name__ == '__main__':
    #1. MAKE DIRECTORIES
    make_dir()
    #2. RESIZE IMAGES AND SAVE TO CORRESPONDING DIRECTORIES
    process_data(512,512)
    #3. Perform colour transformations and down sampling 
    generate_train_data()
    generate_valid_data()
    #4 GENERATE TRAINING AND VALIDATION NUMPY ARRAYS
    y,u,v = generate_train_np()
    y_valid,u_valid,v_valid = generate_validation_np()
    rgb_train = generate_RGB_train_np()
    rgb_valid = generate_RGB_valid_np()
    #5 SAVE DATASET NUMPY ARRAYS TO DIRECTORY
    save_data_as_numpy_arrays(y,u,v,y_valid,u_valid,v_valid,rgb_train,rgb_valid)
