#!/usr/bin/python

from __future__ import print_function
import random
from initGame import initgame
from initNetwork import *
import tensorflow as tf
import numpy as np
from collections import deque
import nn_calc as nc
import math
import sys
import os


IMAGE_SIZE_X = 120 # resolution of the image for the network
IMAGE_SIZE_Y = 120

KERNEL1 = 5
KERNEL2 = 3
KERNEL3 = 2

STRIDE1 = 4
STRIDE2 = 2
STRIDE3 = 1

GAMMA = 0.95 # decay rate of past observations
OBSERVE = 10000 # timesteps to observe before training
#EXPLORE = 10000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
#K = 1 # only select an action every Kth frame, repeat prev for others
#STACK = 1 # number of images stacked to a state
GAME = "Doom"
END = int( 2 * math.pow(10,6) )
#END = 2000
STORE = int( 0.5 * math.pow(10,6) )

def trainNetwork(actions, num_actions, game, s, readout, h_fc1, sess, stack, frame_action, anneal_epsilon, with_depth, evaluate, feedback):
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

    store_path = "logs_stack" + str(stack) + "_frame_action" + str(frame_action) + "_annealing" + str(anneal_epsilon) + "_withDepth" + str(with_depth)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    
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
    
    # create first image
    gray = nc.getGray(game_state)
    
    if with_depth:
        depth = game_state.image_buffer[3,:,:]
        x_t = nc.image_postprocessing_depth(gray, depth, IMAGE_SIZE_Y, IMAGE_SIZE_X)
    else:
        x_t = nc.image_postprocessing(gray, IMAGE_SIZE_Y, IMAGE_SIZE_X)
    
    # stack images
    s_t = nc.create_state(x_t, stack)
        
    # saver for the weights
    saver = tf.train.Saver(max_to_keep=100)
    # start session
    sess.run(tf.initialize_all_variables())
    
    # saving and loading weights
    checkpoint = tf.train.get_checkpoint_state(store_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights, starting from scratch")
        
    # define learning parameters
    if evaluate:
        epsilon = FINAL_EPSILON
        observe = 100
        end = 2000
    else:
        observe = OBSERVE
        epsilon = INITIAL_EPSILON
        end = END
        
    t = 0

    if feedback:
        imgcnt = 0
        maximg = 10
    
    print("************************* Running *************************")
    
    while "pigs" != "fly":
        #print('t:',t)
        # get the Q-values of every action for the current state
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        # the zero makes an array out of the returend matrix (3,1)
        
        if feedback:        
            print("Q-values:", readout_t)        
        
        # choose random action or best action (dependent on epsilon)
        a_t =  [0] * num_actions
        action_index = 0
        if random.random() <= epsilon or t <= observe:
            action_index = random.randrange(num_actions)
            a_t[random.randrange(num_actions)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
            
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / anneal_epsilon
            if epsilon <= FINAL_EPSILON:
                print("Epsilon annealed to", epsilon, "%")
            
        # perform action and create new state (for K frames, train network after that)
        for i in range(0, frame_action):
            
            # run the selected action and observe next state and reward
            r_t = game.make_action(a_t)
            #print("reward:", r_t)
            terminal = game.is_episode_finished()
            
            # new: restart the game if it terminated?
            if game.is_episode_finished():
                game.new_episode()
            
            #get the new game state and the new image
            game_state = game.get_state()
            #x_t1 = game_state.image_buffer[0,:,:]
            #x_t1 = nc.image_postprocessing(x_t1, IMAGE_SIZE_X, IMAGE_SIZE_Y, True)
            gray = nc.getGray(game_state)
            
            if with_depth:
                depth = game_state.image_buffer[3,:,:]
                x_t1 = nc.image_postprocessing_depth(gray, depth, IMAGE_SIZE_Y, IMAGE_SIZE_X)
            else:
                x_t1 = nc.image_postprocessing(gray, IMAGE_SIZE_Y, IMAGE_SIZE_X)
            
            #stack image with the last three images from the old state to create new state
            s_t1 = nc.update_state(s_t, x_t1)
            
            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
                
        if t > observe:
            if t == observe+1:
                print("Observing done")
            
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

        # update the old values
        s_t = s_t1
        t += 1
        
        if feedback:
            if t % 10 == 0 and imgcnt < maximg:
                nc.store_img(x_t1, t)
                imgcnt += 1
        
        # save progress every 10000 iterations
        if t % STORE == 0:
            saver.save(sess, store_path + '/' + GAME + '-dqn', global_step = t)
            print("Saved weights after", t, "steps")
        
        if t == end:
            reward = game.get_total_reward()
            reward = reward/t  
            reward_path = store_path + "/reward.txt"
            reward_file = open(reward_path, 'w')
            reward_file.write(str(reward))
            reward_file.close()            
            
            print("Network reached step", end, ", terminating")
            print("Reward per step:", reward)
            print("************************* Done *************************")
            break

         # print info
#        state = ""
#        if t <= OBSERVE:
#            state = "observe"
#        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
#            state = "explore"
#        else:b
#            state = "train"
#        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))                
    game.close()
      
def main():
#    stack = [1, 3, 5]
#        frame_action = [1, 3]
#        explore_anneal = [0.1, 1, 3] * math.pow(10,6)
#        
#        for i in range(0, len(stack)):
#            for j in range(0, len(frame_action)):
#                for k in range(0, len(explore_anneal)):  

    EVALUATE = True
    FEEDBACK = True
    
      
    stack = int(sys.argv[1])
    frame_action = int(sys.argv[2])
    explore_anneal = int(sys.argv[3])
    with_depth = sys.argv[4]
    if with_depth == "1":
        with_depth = True
    else:
        with_depth = False
        
    if EVALUATE:
        print("Testing results from network with following parameters:")
        print("Stack:", stack, "Frame/Action:", frame_action, "Depth:", with_depth)
    else:
        print("Executing network with following parameters:")
        print("Images stacked together:", stack)
        print("New action will be taken every", frame_action, "frame")
        print("Randomization factor will anneal to", FINAL_EPSILON, "% over", explore_anneal, "steps")
        print("Adding depth image:", with_depth)  
        print("Network will observe for", OBSERVE, "steps before calibrating")
        print("Weights will be saved every", STORE, "steps")
        print("Network will calibrate for a maximum of", END, "steps")
    
    actions, num_actions, game = initgame()
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork(num_actions, stack, IMAGE_SIZE_Y/2, IMAGE_SIZE_X, KERNEL1, STRIDE1, KERNEL2, STRIDE2, KERNEL3, STRIDE3)
    trainNetwork(actions, num_actions, game, s, readout, h_fc1, sess, stack, frame_action, explore_anneal, with_depth, EVALUATE, FEEDBACK)
    
if __name__ == "__main__":
    main()