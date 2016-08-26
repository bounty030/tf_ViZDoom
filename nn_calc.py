# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:38:13 2016
This file makes all the calculations with openCV
@author: martin
"""

import math
import cv2
import numpy as np
import os

STORED = False

# calculates the dimensions of the feature maps after a kernel with stride x & y
def post_kernel(dim_x, dim_y, stride):
    
    #withPad_x = dim_x + math.floor(2*(kernel_size/2))
    #withPad_y = dim_y + math.floor(2*(kernel_size/2))
    
    feature_x = math.ceil(dim_x/stride)
    feature_y = math.ceil(dim_y/stride)
    
    #print feature_x
    #print feature_y
    
    return feature_x, feature_y

# calcualtes the dimensions after pooling
def post_pool(dim_x, dim_y):
    
    pool_x = dim_x/2
    pool_y = dim_y/2
    
    #print pool_x
    #print pool_y
    
    return pool_x, pool_y

# simple postprocessing
def image_postprocessing(img, t_size_y, t_size_x, feedback, t):
	
	# resize image
    img = cv2.resize(img, (t_size_y, t_size_x))
    
    # cut image
    img = img[t_size_y/2-1:-1,:]
    
    # filter image
    ret,img = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
    
    #store if flag is set
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_filter.png', img)
    
    return img
    
# add image and depth image
# will store all stage of processing if flag for feedback is set
def image_postprocessing_depth(gray, depth, t_size_y, t_size_x, feedback, t):

    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_gray_0_input.png', gray) 
        cv2.imwrite('feedback/image_' + str(t) + '_depth_0_input.png', gray) 
    
    # resize normal image
    gray = cv2.resize(gray, (t_size_y, t_size_x))
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_gray_1_resize.png', gray)
    
    # resize depth image
    depth = cv2.resize(depth, (t_size_y, t_size_x))
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_depth_1_resize.png', depth)
	
	# cut normal image
    gray = gray[t_size_y/2-1:-1,:]
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_gray_2_cut.png', gray)
	
	# cut depth image
    depth = depth[t_size_y/2-1:-1,:]
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_depth_2_cut.png', depth)
    
    # threshold filter for the grayscale image
    ret,gray = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
    
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_gray_3_flt.png', gray)
    
    
    # custom filter for the depth image
    depth = cv2.bitwise_not(depth)
    ret, depth = cv2.threshold(depth,165,255,cv2.THRESH_TOZERO)
    
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_depth_3_flt_inv.png', depth)
    
    height, width = depth.shape

    # subtract lowest gray-value
    minval = np.min(depth[np.nonzero(depth)])
    depth[np.nonzero(depth)] -= minval
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_depth_4_off.png', depth)
    
    # return the added image
    result = cv2.add(gray,depth)
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_final.png', result)
    
    return result

# calculates the gray-scale image from ViZDoom
def getGray(game_state):
    red = game_state.image_buffer[0,:,:]
    green = game_state.image_buffer[1,:,:]
    blue = game_state.image_buffer[2,:,:]
    gray = cv2.merge((blue,green,red))
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return gray  

# calculates the color-image from ViZDoom
def getColor(game_state):
    red = game_state.image_buffer[0,:,:]
    green = game_state.image_buffer[1,:,:]
    blue = game_state.image_buffer[2,:,:]
    color = cv2.merge((blue,green,red))
    return color        
    
# adds an image to the state variable
def update_state(state, img):
    img = np.reshape(img, (len(img), len(img[0]), 1))
    return np.append(img, state[:,:,1:], axis = 2)
  
# stacks one image multiple times for the first stack  
def create_state(img, stack):
    img = np.reshape(img, (len(img), len(img[0]), 1))
    state = img
    for i in range(1, stack):
        state = np.append(img, state, axis=2)
    return state

# returns t in four digits
def get_t(t):
    
    if t < 10:
        return "000" + str(t)
    if t < 100:
        return "00" + str(t)
    if t < 1000:
        return "0" + str(t)
    return str(t) 
    
def store_img(img):
    cv2.imwrite('image.png', img)
    
def store_img(img, add):
    name  = 'image_' + str(add) + '.png'
    cv2.imwrite(name, img)
    
def store_img(img, add, path):
    name  = 'image_' + str(add) + '.png'
    cv2.imwrite(os.path.join(path, name), img)
