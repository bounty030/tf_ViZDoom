# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:38:13 2016

@author: martin
"""

import math
import cv2
import numpy as np
import os

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
    
def image_postprocessing(img, t_size_y, t_size_x, feedback, t):
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '.png', img)
    img = cv2.resize(img, (t_size_y, t_size_x))
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_1.png', img)
    img = img[(t_size_y/2 - 20/2):(t_size_y/2 + 20/2),:]
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_2.png', img)
    
    ret,img = cv2.threshold(img,18,255,cv2.THRESH_BINARY_INV)
    
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_3.png', img)
    
    return img
    
# add image and depth image
def image_postprocessing_depth(gray, depth, t_size_y, t_size_x, feedback, t):
    
    # resize and cut images
    gray = cv2.resize(gray, (t_size_y, t_size_x))
    depth = cv2.resize(depth, (t_size_y, t_size_x))
    gray = gray[t_size_y/2-1:-1,:]
    depth = depth[t_size_y/2-1:-1,:]
    
    color = gray
    color = cv2.cvtColor(gray, color, cv2.CV_GRAY2RGB);
    
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_gray.png', gray)
        cv2.imwrite('feedback/image_' + str(t) + '_color.png', color)
        cv2.imwrite('feedback/image_' + str(t) + '_depth.png', depth)
    
    # threshold filter for the grayscale image
    ret,gray = cv2.threshold(gray,160,255,cv2.THRESH_BINARY)
    
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_gray_flt.png', gray)
    
    
    # custom filter for the depth image
    depth = cv2.bitwise_not(depth)
    ret, depth = cv2.threshold(depth,165,255,cv2.THRESH_TOZERO)
    
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_depth_1.png', depth)
    
    height, width = depth.shape

    # subtract lowest gray-value
    minval = np.min(depth[np.nonzero(depth)])
    depth[np.nonzero(depth)] -= minval
    if feedback:
        cv2.imwrite('feedback/image_' + str(t) + '_depth_2.png', depth)
    
    # return the added image
    result = cv2.add(gray,depth)
    #if feedback:
    #    cv2.imwrite('feedback/image_' + str(t) + '_filter.png', result)
    
    return result

def getGray(game_state):
    red = game_state.image_buffer[0,:,:]
    green = game_state.image_buffer[1,:,:]
    blue = game_state.image_buffer[2,:,:]
    gray = cv2.merge((blue,green,red))
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    return gray    

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
    
def store_img(img):
    cv2.imwrite('image.png', img)
    
def store_img(img, add):
    name  = 'image_' + str(add) + '.png'
    cv2.imwrite(name, img)
    
def store_img(img, add, path):
    name  = 'image_' + str(add) + '.png'
    cv2.imwrite(os.path.join(path, name), img)
    
