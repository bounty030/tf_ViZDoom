# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:38:13 2016

@author: martin
"""

import math
import cv2
import numpy as np

def post_kernel(dim_x, dim_y, stride):
    
    #withPad_x = dim_x + math.floor(2*(kernel_size/2))
    #withPad_y = dim_y + math.floor(2*(kernel_size/2))
    
    feature_x = math.ceil(dim_x/stride)
    feature_y = math.ceil(dim_y/stride)
    
    #print feature_x
    #print feature_y
    
    return feature_x, feature_y
    
def post_pool(dim_x, dim_y):
    
    pool_x = dim_x/2
    pool_y = dim_y/2
    
    #print pool_x
    #print pool_y
    
    return pool_x, pool_y
    
def image_postprocessing(img, t_size_x, t_size_y):
    res = cv2.resize(img, (t_size_x, t_size_y))
    #return res
    
    #current temp modification
    MIDDLE_STRIPE_SIZE = 20
    return res[(t_size_y/2 - MIDDLE_STRIPE_SIZE/2):(t_size_y/2 + MIDDLE_STRIPE_SIZE/2),:]
    
def update_state(state, img):
    img = np.reshape(img, (len(img), len(img[0]), 1))
    return np.append(img, state[:,:,1:], axis = 2)
    
def create_state(img, stack):
    img = np.reshape(img, (len(img), len(img[0]), 1))
    state = img
    for i in range(1, stack):
        state = np.append(img, state, axis=2)
    return state
    