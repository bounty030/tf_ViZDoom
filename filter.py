#!/usr/bin/python
#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################

from __future__ import print_function
#import matplotlib 
#from matplotlib import pyplot as plt
import random
from time import sleep
from time import time
from vizdoom import *
import cv2
import numpy as np
import sys
import nn_calc as nc
#from PyQt5 import QtCore, QtGui, QtWidgets


# Run this many episodes
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will work. Note that the most recent changes will add to previous ones.
#game.load_config("../../examples/config/basic.cfg")

vizdoom_path = "../ViZDoom"

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
game.load_config(vizdoom_path + "/examples/config/basic.cfg")

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
game.set_vizdoom_path(vizdoom_path + "/bin/vizdoom")

# Sets path to doom2 iwad resource file which contains the actual doom game. Default is "./doom2.wad".
game.set_doom_game_path(vizdoom_path + "/scenarios/freedoom2.wad")
#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.

# Sets path to additional resources iwad file which is basically your scenario iwad.
# If not specified default doom2 maps will be used and it's pretty much useles... unless you want to play doom.
# game.set_doom_scenario_path(vizdoom_path + "/scenarios/health_gathering.wad")
game.set_doom_scenario_path(vizdoom_path + "/scenarios/basic.wad")

# Sets map to start (scenario .wad files can contain many maps).
game.set_doom_map("map01")

# Sets resolution. Default is 320X240
game.set_screen_resolution(ScreenResolution.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.CRCGCB)

# Sets other rendering options
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(False)
game.set_render_decals(False)
game.set_render_particles(False)

# Adds buttons that will be allowed. 
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(200)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(True)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(False)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Initialize the game. Further configuration won't take any effect from now on.

IMAGE_SIZE = 80 # resollution of the image for the network (square)
POST_PROCESS = True
FILTER_ALL = False
# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.	
actions = [[True,False,False],[False,True,False],[False,False,True]]
game.init()
episodes = 1

game.new_episode()
game_state = game.get_state()

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup. 
sleep_time = 0.05
for i in range(episodes):
   # s = game.get_state()
   # r = game.make_action(choice(actions))
    red = game_state.image_buffer[0,:,:]
    green = game_state.image_buffer[1,:,:]
    blue = game_state.image_buffer[2,:,:]
    if sleep_time>0:
        sleep(sleep_time)
        
#    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    game.close()
    
target = red    
    
original = cv2.merge((blue,green,red))    
    
cv2.imwrite('filter/red.png',red)
cv2.imwrite('filter/green.png',green)
cv2.imwrite('filter/blue.png',blue)
cv2.imwrite('filter/original.png', original)

current = nc.image_postprocessing(red, IMAGE_SIZE, IMAGE_SIZE, True)
cv2.imwrite('filter/current.png',current)
#cv2.imshow('image for the network',current)
#cv2.waitKey()



ret,thresh1 = cv2.threshold(target,100,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(target,80,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(target,80,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(target,80,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(target,80,255,cv2.THRESH_TOZERO_INV)
    
th1 = cv2.adaptiveThreshold(target,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,5,2)
    
th2 = cv2.adaptiveThreshold(target,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,5,2)
    
 # Otsu's thresholding
ret2,th3 = cv2.threshold(target,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
 # Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(target,(5,5),0)
ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
median = cv2.medianBlur(target,5)
ret3,th5 = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    
bilateral = cv2.bilateralFilter(target,9,10,10)
ret3,th6 = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 


#img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  
if FILTER_ALL:
    if POST_PROCESS:
        cv2.imwrite('filter/tresh1.png',nc.image_postprocessing(thresh1, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/tresh2.png',nc.image_postprocessing(thresh2, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/tresh3.png',nc.image_postprocessing(thresh3, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/tresh4.png',nc.image_postprocessing(thresh4, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/tresh5.png',nc.image_postprocessing(thresh5, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/Atresh1.png',nc.image_postprocessing(th1, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/Atresh2.png',nc.image_postprocessing(th2, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/Atresh3.png',nc.image_postprocessing(th3, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/Atresh4.png',nc.image_postprocessing(th4, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/Atresh5.png',nc.image_postprocessing(th5, IMAGE_SIZE, IMAGE_SIZE, False))
        cv2.imwrite('filter/Atresh6.png',nc.image_postprocessing(th6, IMAGE_SIZE, IMAGE_SIZE, False))
    else:
        cv2.imwrite('filter/tresh1.png',thresh1)
        cv2.imwrite('filter/tresh2.png',thresh2)
        cv2.imwrite('filter/tresh3.png',thresh3)
        cv2.imwrite('filter/tresh4.png',thresh4)
        cv2.imwrite('filter/tresh5.png',thresh5)
        cv2.imwrite('filter/Atresh1.png',th1)
        cv2.imwrite('filter/Atresh2.png',th2)
        cv2.imwrite('filter/Atresh3.png',th3)
        cv2.imwrite('filter/Atresh4.png',th4)
        cv2.imwrite('filter/Atresh5.png',th5)
        cv2.imwrite('filter/Atresh6.png',th6)
        
print("Done")