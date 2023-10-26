import math
import numpy as np

#Calculate PSNR

def MSE_PSNR(imageA, imageB):
    MSE = np.sum((imageA - imageB) ** 2)
    MSE /= (imageA.shape[0] * imageA.shape[1])
    PSNR = 10*math.log10(255**2/MSE)
    print("PSNR = ",PSNR)
    return MSE, PSNR