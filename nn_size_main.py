# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 19:25:21 2016

@author: martin
"""

import math

DIM_X = 80
DIM_Y = 20

KERNEL1 = 4
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
    
if __name__ == "__main__":
    main()