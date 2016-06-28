# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 19:25:21 2016

@author: martin
"""

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt

DIM_X = 100
DIM_Y = 50

KERNEL1 = 5
KERNEL2 = 3
KERNEL3 = 2

STRIDE1 = 4
STRIDE2 = 2
STRIDE3 = 1

def post_kernel(dim_x, dim_y, stride):
    
    #withPad_x = dim_x + math.floor(2*(kernel_size/2))
    #withPad_y = dim_y + math.floor(2*(kernel_size/2))
    
    feature_x = math.ceil(dim_x/stride)
    feature_y = math.ceil(dim_y/stride)
    
    #print feature_x
    #print feature_y
    
    return feature_x, feature_y
    
def post_pool(dim_x, dim_y):
    
    pool_x = math.ceil(dim_x/2)
    pool_y = math.ceil(dim_y/2)
    
    #print pool_x
    #print pool_y
    
    return pool_x, pool_y
    
def main():
    dim1x, dim1y = post_kernel(DIM_X, DIM_Y, STRIDE1)
    dim1x, dim1y = post_pool(dim1x, dim1y)
    print "1. layer size: ", dim1x, "/", dim1y
    
    dim2x, dim2y = post_kernel(dim1x, dim1y, STRIDE2)
    print "2. layer size: ", dim2x, "/", dim2y
    
    dim3x, dim3y = post_kernel(dim2x, dim2y, STRIDE3)
    print "2. layer size: ", dim3x, "/", dim3y
    
    # example kernel
    img = cv2.imread('image.png')
    
    # define kernel here!
    kernel = (np.zeros((KERNEL1,KERNEL1))) * -1
    #kernel[0,0] = 1
    #kernel[KERNEL1-1,0] = -1
    #kernel[0,KERNEL1-1] = -1    
    #kernel[KERNEL1-1,KERNEL1-1] = 1
    
    #kernel[(KERNEL1-1)/2, (KERNEL1-1)/2] = KERNEL1*KERNEL1-1
    
    kernel[0,0] = -1
    kernel[1,0] = -1
    kernel[2,0] = -1
    kernel[0,2] = 1
    kernel[1,2] = 1
    kernel[2,2] = 1
    

    # convolution
    dst = cv2.filter2D(img,-1,kernel)
    
    #plot
    plt.subplot(121),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(dst),plt.title('Edge')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
if __name__ == "__main__":
    main()