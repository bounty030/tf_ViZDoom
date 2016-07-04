# -*- coding: utf-8 -*-
import tensorflow as tf
import nn_calc as nc

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork(num_actions, stack, image_size_x, image_size_y, kernel_size1, stride1, kernel_size2, stride2, kernel_size3, stride3):
    # calculate dimensions of network
    dim_x1, dim_y1 = nc.post_kernel(image_size_x, image_size_y, stride1)
    dim_x1, dim_y1 = nc.post_pool(dim_x1, dim_y1)
    dim_x2, dim_y2 = nc.post_kernel(dim_x1, dim_y1, stride2)
    dim_x3, dim_y3 = nc.post_kernel(dim_x2, dim_y2, stride3)
    
    # network weights
    W_conv1 = weight_variable([kernel_size1, kernel_size1, stack, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([kernel_size2, kernel_size2, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([kernel_size3, kernel_size3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([int(dim_x3*dim_y3*64), 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, num_actions])
    b_fc2 = bias_variable([num_actions])

    # input layer
    s = tf.placeholder("float", [None, image_size_x, image_size_y, stack])

    # first hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, stride1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)
    
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, stride3) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)
    
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, int(dim_x3*dim_y3*64)])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1