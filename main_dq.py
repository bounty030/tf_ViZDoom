#!/usr/bin/python

from __future__ import print_function
from time import sleep
import random
from initGame import initgame
from initNetwork import *
import tensorflow as tf
import numpy as np
from collections import deque

import cv2

TEST = False # Testrun
IMAGE_SIZE = 80 # resollution of the image for the network (square)
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 1000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 590000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
GAME = "Doom"



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
    # todo check vizdoom replay memory
    
    # do nothing from deep_q
    # i would say we just start with an action
    # get first image
    # index zero, because buffer is of form (n,y,x)
    # n -> stack?
    x_t = game_state.image_buffer[0,:,:]
    x_t = cv2.transpose(x_t)
    x_t = cv2.resize(x_t, (IMAGE_SIZE, IMAGE_SIZE))
    
    # stack images
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
    # tensorflow kram den ich nicht verstehe
    saver = tf.train.Saver()
    #start session
    sess.run(tf.initialize_all_variables())
    
    # saving and loading weights it seems
    checkpoint = tf.train.get_checkpoint_state("logs")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
        
    #start the learning
    epsilon = INITIAL_EPSILON
    t = 0
    # extra fuer tim so gelassen!
    while "pigs" != "fly":
        # get the Q-values of every action for the current state
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        # the zero makes an array out of the returend matrix (3,1)
        
        # choose random action or best action (dependent on epsilon)
        #a_t = np.zeros([num_actions])
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
            
        # perfoem action and create new state (for K frames, train network after that)
        for i in range(0, K):
            # run the selected action and observe next state and reward
            #print("a_t: ",a_t)
            r_t = game.make_action(a_t)
            
            terminal = game.is_episode_finished()
            #print("terminal:", terminal)
            
            # new: restart the game if it terminated?
            if game.is_episode_finished():
                game.new_episode()
            
            #get the new game state and the new image
            game_state = game.get_state()
            x_t1 = game_state.image_buffer[0,:,:]
            x_t1 = cv2.transpose(x_t1)
            x_t1 = cv2.resize(x_t1, (IMAGE_SIZE, IMAGE_SIZE))
            
            #stack image with the last three images from the old state to create new state
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            
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

        # update the old values
        s_t = s_t1
        t += 1
            
        if (TEST and t>5):
            game.close()
            break
        
        # save progress every 10000 iterations
        if t % 3000 == 0:
            saver.save(sess, 'logs/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))

        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''
            
            
    
#    
#    
#    episodes = 20
#
#    # Sets time that will pause the engine after each action.
#    # Without this everything would go too fast for you to keep track of what's happening.
#    # 0.05 is quite arbitrary, nice to watch with my hardware setup. 
#    sleep_time = 0.028
#
#    for i in range(episodes):
#        print("Episode #" + str(i+1))
#
#    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
#        game.new_episode()
#
#        while not game.is_episode_finished():
#
#        # Gets the state
#            s = game.get_state()
#
#        # Makes a random action and get remember reward.
#            r = game.make_action(random.choice(actions))
#
#        # Prints state's game variables. Printing the image is quite pointless.
#            print("State #" + str(s.number))
#            print("Game variables:", s.game_variables[0])
#            print("Reward:", r)
#            print("=====================")
#
#            if sleep_time>0:
#                sleep(sleep_time)
#
#    # Check how the episode went.
#        print("Episode finished.")
#        print("total reward:", game.get_total_reward())
#        print("************************")
#        game.close()
    
def main():
    actions, num_actions, game = initgame()
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork(num_actions)
    trainNetwork(actions, num_actions, game, s, readout, h_fc1, sess)
    
if __name__ == "__main__":
    main()