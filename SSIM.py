import numpy as np
def SSIM(image_A,image_B):
    #K_1,K_2 from https://arxiv.org/pdf/2006.13846.pdf
    K_1 = 0.01
    K_2 = 0.03
    L = 255 #8 bit image 0<->255
    mean_A = np.mean(image_A)
    mean_B = np.mean(image_B)
    var_A  = np.var(image_A)
    var_B = np.var(image_B)
    #need to flatten() to turn each image into a feature vector
    #take element at [0][1] because we want covar AB, 
    #covar_AB is a 2x2 matrix: 
    # var(A), cov(A,B)
    # cov(B,A), var(B)
    covar_AB = np.cov(image_A.flatten(),image_B.flatten())[0][1]
    C_1 = (K_1*L)**2
    C_2 = (K_2*L)**2
    numerator = (2*mean_A*mean_B + C_1)*(2*covar_AB + C_2)
    denominator = (mean_A**2+mean_B**2+C_1)*(var_A+var_B+C_2)
    
    SSIM = numerator/denominator
    return SSIM
    