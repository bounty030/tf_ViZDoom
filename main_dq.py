#!/usr/bin/python

from __future__ import print_function
import random
from initGame import initgame
from initNetwork import *
import tensorflow as tf
import numpy as np
from collections import deque
import nn_calc as nc
import os
import time


IMAGE_SIZE_X = 80 # resolution of the image for the network
IMAGE_SIZE_Y = 80

KERNEL1 = 4
KERNEL2 = 3
KERNEL3 = 2

STRIDE1 = 4
STRIDE2 = 2
STRIDE3 = 1

GAMMA = 0.95 # decay rate of past observations
OBSERVE = 100 # timesteps to observe before training
EXPLORE = 500000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
STACK = 4 # number of images stacked to a state
GAME = "Doom"
FEEDBACK = False
END = 100000
SLEEPTIME = 0#0.028

def writeConfig(path):
    config_file = open(path, 'a')
    config_file.write("Image Size: " + str(IMAGE_SIZE_X) + "x" + str(IMAGE_SIZE_Y) + "\n")
    config_file.write("Kernel1- " + "Size: " + str(KERNEL1) + "   Stride:" + str(STRIDE1) + "\n")
    config_file.write("Kernel2- " + "Size: " + str(KERNEL2) + "   Stride:" + str(STRIDE2) + "\n")
    config_file.write("Kernel3- " + "Size: " + str(KERNEL3) + "   Stride:" + str(STRIDE3) + "\n\n")
    config_file.write("Gamma: " + str(GAMMA) + "\n")
    config_file.write("Observe: " + str(OBSERVE) + "\n")
    config_file.write("Explore: " + str(EXPLORE) + "\n")
    config_file.write("Final epsilon: " + str(FINAL_EPSILON) + "\n")
    config_file.write("Replay Memory: " + str(REPLAY_MEMORY) + "\n")
    config_file.write("Batch Size: " + str(BATCH) + "\n")
    config_file.write("Choose new action every : " + str(K) + " frames" + "\n")
    config_file.write("Image stacking: " + str(STACK) + " images" + "\n")
    config_file.write("Total number of actions: " + str(END) + "\n\n\n")
    config_file.close()  


def trainNetwork(actions, num_actions, game, s, readout, h_fc1, sess):
#==============================================================================
#
# variables:
# actions   array that contains the defined arrays for each action
# num_actions   amount of actions
# game      doom-game
# s         input-layer (80,80,?)-image
# readout   result of the network (Q), last layer of the network
# h_fc1     second-last layer of the network
# sess      tensorflow-session
#  
#==============================================================================
    t = 0  

    t_old_save = 0
    monster_count = 0
    reward_all = 0
    
    if FEEDBACK:
        imgcnt = 0
        maximg = 100
        
        feedback_path = "feedback"
        if not os.path.exists(feedback_path):
            os.makedirs(feedback_path)
            os.makedirs(feedback_path + "/forVideo")
        qfile_path = feedback_path + "/qfile.txt"    
    
    store_path = "logs/"
    if not os.path.exists(store_path):
        os.makedirs(store_path)
        
    reward_path = store_path + "BasicReward_" + "Stack" + str(STACK) + "-Kernel1Stride" + str(STRIDE1) + "-" + str(END) + "Actions" + ".txt"
    
    writeConfig(reward_path)
    
    #tensorflow variable for the actions
    a = tf.placeholder("float", [None, num_actions])
    #tensorflow variable for the target in the cost function
    y = tf.placeholder("float", [None])
    #multiply the action with the result of our network => Q(s,a)
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    
    #cost function and gradient
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    # open up a game state to communicate with emulator
    game.new_episode()
    # get the game state
    game_state = game.get_state()
    
    # store the previous observations in replay memory
    D = deque()
    
    # get first image
    # index zero, because buffer is of form (n,y,x)
    # n -> color? 
    image = game_state.image_buffer[0,:,:]
    x_t = nc.image_postprocessing(image, IMAGE_SIZE_Y, IMAGE_SIZE_X, False, t)
    # stack images
    s_t = nc.create_state(x_t, STACK)
        
    # saver for the weights
    saver = tf.train.Saver()
    # start session
    sess.run(tf.initialize_all_variables())
    
    # saving and loading weights
    checkpoint = tf.train.get_checkpoint_state("logs")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights, starting from scratch")
        
    # define learning parameters
    epsilon = INITIAL_EPSILON
    
    print("Observing for", OBSERVE, "turns, calibrating afterwards")
    
    start_time = time.time()   
    
    while "pigs" != "fly":
        # get the Q-values of every action for the current state
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        # the zero makes an array out of the returend matrix (3,1)
        
        # choose random action or best action (dependent on epsilon)
        a_t =  [0] * num_actions
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(num_actions)
            a_t[random.randrange(num_actions)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
            
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
        # perform action and create new state (for K frames, train network after that)
        for i in range(0, K):
            
            # run the selected action and observe next state and reward
            r_t = game.make_action(a_t)
            reward_all += r_t
            if r_t > 10:
                monster_count += 1
            
            terminal = game.is_episode_finished()
            
            # new: restart the game if it terminated?
            if game.is_episode_finished():
                game.new_episode()
            
            #get the new game state and the new image
            game_state = game.get_state()
            image = game_state.image_buffer[0,:,:]
            if FEEDBACK:
                if t % 1 == 0 and imgcnt < maximg:
                    x_t1 = nc.image_postprocessing(image, IMAGE_SIZE_Y, IMAGE_SIZE_X, FEEDBACK, t)
                else:
                    x_t1 = nc.image_postprocessing(image, IMAGE_SIZE_Y, IMAGE_SIZE_X, False, t)
            else:
                x_t1 = nc.image_postprocessing(image, IMAGE_SIZE_Y, IMAGE_SIZE_X, False, t)
            
            if FEEDBACK:
                color = nc.getColor(game_state)
                nc.store_img(color, str(t), feedback_path + "/forVideo")
            
            #stack image with the last three images from the old state to create new state
            s_t1 = nc.update_state(s_t, x_t1)
            
            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
                
        if t > OBSERVE:
            
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_batch   = [d[0] for d in minibatch]
            a_batch   = [d[1] for d in minibatch]
            r_batch   = [d[2] for d in minibatch]
            s1_batch  = [d[3] for d in minibatch]
            
            # feed our network with the future state and predict the Q-values
            predictedQ_batch = readout.eval(feed_dict = {s : s1_batch})
            
            # target variable in cost function
            y_batch = []
            
            # calculate y for all transitions in the minibatch
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(predictedQ_batch[i]))
                    
            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_batch})
                
            if FEEDBACK:
                print("t:", t, "Monster shot", monster_count)
                
                #todo store q-value and image every x steps
                if t % 1 == 0 and imgcnt < maximg:
                    #nc.store_img(image, t, feedback_path)
                    imgcnt += 1
                    
                    #and store the corresponding q-values
                    qfile = open(qfile_path, 'a')
                    qfile.write(str(t) + ": Q-Values:" + str(readout_t) + "\n")
                    qfile.close()  
                    
                if t == END:
                    print("terminating")
                    qfile = open(qfile_path, 'a')
                    qfile.write("***** done *****\n")
                    qfile.close()

                    reward_p_turn = reward_all / (t-t_old_save)                  
                    
                    reward_file = open(reward_path, 'a')
                    reward_file.write(str(t) + ": reward " + str(reward_p_turn) + ", monster " + str(monster_count) + "\n")
                    reward_file.close() 
                    game.close()
                    break

        # update the old values
        s_t = s_t1
        t += 1
        
        if SLEEPTIME > 0:
            time.sleep(SLEEPTIME)
                      
        # save progress every 10000 iterations
        if t % 100 == 0:
            #saver.save(sess, 'logs/' + GAME + '-dqn', global_step = t)

            current_time = time.time() - start_time
            reward_p_turn = reward_all / (t-t_old_save)
                
            reward_file = open(reward_path, 'a')
            reward_file.write(str(t) + ":\n reward " + str(reward_p_turn) + ", time: " + str(current_time) + ", monster " + str(monster_count) + "\n")
            reward_file.close()     
            
            monster_count = 0
            reward_all = 0
            t_old_save = t
            
            print("Saved weights after", t, "steps")
        
      
def main():
    actions, num_actions, game = initgame(FEEDBACK)
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork(num_actions, STACK, 20, IMAGE_SIZE_X, KERNEL1, STRIDE1, KERNEL2, STRIDE2, KERNEL3, STRIDE3)
    trainNetwork(actions, num_actions, game, s, readout, h_fc1, sess)
    
if __name__ == "__main__":
    main()
