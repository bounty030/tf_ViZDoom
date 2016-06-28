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

def initgame():
# Run this many episodes
  game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will work. Note that the most recent changes will add to previous ones.
#game.load_config("../../examples/config/basic.cfg")

  vizdoom_path = "../ViZDoom"

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
  game.load_config(vizdoom_path + "/examples/config/health_gathering.cfg")

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
  game.set_vizdoom_path(vizdoom_path + "/bin/vizdoom")

# Sets path to doom2 iwad resource file which contains the actual doom game. Default is "./doom2.wad".
  game.set_doom_game_path(vizdoom_path + "/scenarios/freedoom2.wad")
#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.

# Sets path to additional resources iwad file which is basically your scenario iwad.
# If not specified default doom2 maps will be used and it's pretty much useles... unless you want to play doom.
  game.set_doom_scenario_path(vizdoom_path + "/scenarios/health_gathering.wad")

# Sets map to start (scenario .wad files can contain many maps).
  game.set_doom_map("map01")

# Sets resolution. Default is 320X240
  game.set_screen_resolution(ScreenResolution.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
  game.set_screen_format(ScreenFormat.CRCGCBDB)

# Sets other rendering options
  game.set_render_hud(False)
  game.set_render_crosshair(False)
  game.set_render_weapon(True)
  game.set_render_decals(False)
  game.set_render_particles(False)

# Adds buttons that will be allowed. 
  game.add_available_button(Button.TURN_LEFT)
  game.add_available_button(Button.TURN_RIGHT)
  game.add_available_button(Button.MOVE_FORWARD)

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



# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.	
  actions = [[True,False,False],[False,True,False],[False,False,True],[True,False,True],[False,True,True]]
  game.init()
  episodes = 120
  #episode_timeout = 80000
  game.new_episode()
  game_state = game.get_state()

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup. 
  sleep_time = 0.05
  for i in range(episodes):
    if i < 80:
        r = game.make_action([False,False,True])
    else:
        r = game.make_action([True,False,False])
        
    print ('writing image')
    game_state = game.get_state()
    x =game_state.image_buffer
    red = game_state.image_buffer[0,:,:]
    green = game_state.image_buffer[1,:,:]
    blue = game_state.image_buffer[2,:,:]
    depth = game_state.image_buffer[3,:,:]
    
    if sleep_time>0:
        sleep(sleep_time)
        print("Episode #" + str(i+1))

  game.close()
    
  return red, green, blue, depth

def filtering(red, green, blue, depth):

  IMAGE_SIZE = 200 # resolution of the image for the network (square)

  #Create original image
  original = cv2.merge((blue,green,red))  
  
  original = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
  original = cv2.resize(original, (IMAGE_SIZE, IMAGE_SIZE))
  cv2.imwrite('original/original.png',original)

  #Resize
  red = cv2.resize(red, (IMAGE_SIZE, IMAGE_SIZE))    
  blue = cv2.resize(blue, (IMAGE_SIZE, IMAGE_SIZE))    
  green = cv2.resize(green, (IMAGE_SIZE, IMAGE_SIZE))  
  depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE))  

  #Cut Stripe
  #MIDDLE_STRIPE_SIZE = 20
  #red = red[(IMAGE_SIZE/2 - MIDDLE_STRIPE_SIZE/2):(IMAGE_SIZE/2 + MIDDLE_STRIPE_SIZE/2),:]
  #blue = blue[(IMAGE_SIZE/2 - MIDDLE_STRIPE_SIZE/2):(IMAGE_SIZE/2 + MIDDLE_STRIPE_SIZE/2),:]
  #green = green[(IMAGE_SIZE/2 - MIDDLE_STRIPE_SIZE/2):(IMAGE_SIZE/2 + MIDDLE_STRIPE_SIZE/2),:]

  #Write RGB channel images
  cv2.imwrite('original/red.png',red)
  cv2.imwrite('original/green.png',green)
  cv2.imwrite('original/blue.png',blue)
  cv2.imwrite('original/depth.png',depth)
  depth2 = cv2.bitwise_not(depth)
  cv2.imwrite('original/depth_inv.png',depth2)
  #ret, depth2 = cv2.threshold(depth2,180,255,cv2.THRESH_TRUNC)
  ret, depth2 = cv2.threshold(depth2,165,255,cv2.THRESH_TOZERO)
  cv2.imwrite('original/depth_filt.png',depth2)
  
  height, width = depth2.shape
  lowest = 255
  for i in range(0, height):
      for j in range(0, width):
         if depth2[i,j] < lowest and depth2[i,j] != 0:
            lowest = depth2[i,j]

  for i in range(0, height):
      for j in range(0, width):
          if depth2[i,j] != 0:
            depth2[i,j] = depth2[i,j] - lowest
                  
  cv2.imwrite('original/depth_filt_final.png',depth2)
  
  
  
  #MIDDLE_STRIPE_SIZE = 20
  #red = red[(IMAGE_SIZE/2 - IMAGE_SIZE/2):(IMAGE_SIZE/2 + IMAGE_SIZE/2),:]
  #cv2.imwrite('original/inputred.png',red)

  #green = green[(IMAGE_SIZE/2 - IMAGE_SIZE/2):(IMAGE_SIZE/2 + IMAGE_SIZE/2),:]
  #cv2.imwrite('original/inputgreen.png',green)

  #blue = blue[(IMAGE_SIZE/2 - IMAGE_SIZE/2):(IMAGE_SIZE/2 + IMAGE_SIZE/2),:]
  #cv2.imwrite('original/inputblue.png',blue)

  for x in range(0, 5):
    if x == 0:
      color = red
      textcolor = "red"
      folder = "red/"
    if x == 1:
      color = green
      textcolor = "green"
      folder = "green/"
    if x == 2:
      color = blue
      textcolor = "blue"
      folder = "blue/"
    if x == 3:
      color = depth
      textcolor = "depth"
      folder = "depth/"
    if x == 4:
      color = original
      textcolor = "merged"
      folder = "merged/"
          

    ret,thresh1 = cv2.threshold(color,120,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(color,130,255,cv2.THRESH_BINARY)
    ret,thresh3 = cv2.threshold(color,140,255,cv2.THRESH_BINARY)
    ret,thresh4 = cv2.threshold(color,150,255,cv2.THRESH_BINARY)
    ret,thresh5 = cv2.threshold(color,160,255,cv2.THRESH_BINARY)
    th1 = cv2.adaptiveThreshold(color,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,5,2)   
    th2 = cv2.adaptiveThreshold(color,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,5,2)  
 # Otsu's thresholding
    ret2,th3 = cv2.threshold(color,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
 # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(color,(5,5),0)
    ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    median = cv2.medianBlur(color,5)
    ret3,th5 = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    bilateral = cv2.bilateralFilter(color,9,10,10)
    ret3,th6 = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    cv2.imwrite(folder + 'tresh1.png', thresh1)
    cv2.imwrite(folder + 'tresh2.png', thresh2)
    cv2.imwrite(folder + 'tresh3.png' , thresh3)
    cv2.imwrite(folder + 'tresh4.png' , thresh4)
    cv2.imwrite(folder + 'tresh5.png', thresh5)
    cv2.imwrite(folder + 'tresh6.png', th1)
    cv2.imwrite(folder + 'tresh7.png', th2) #depth!!!
    cv2.imwrite(folder + 'tresh8.png', th3)
    cv2.imwrite(folder + 'tresh9.png', th4)
    cv2.imwrite(folder + 'tresh10.png', th5)
    cv2.imwrite(folder + 'tresh11.png', th6)

def merging():
#==============================================================================
#   img1 = cv2.imread('depth/tresh7.png') 
#   img1 = cv2.bitwise_not(img1)
#   img2 = cv2.imread('blue/tresh1.png')
#   result = cv2.add(img1,img2)
#   cv2.imwrite('result.png',result)
#   cv2.imwrite('depth.png',img1)
#   cv2.imwrite('color.png',img2)
#==============================================================================
  img1 = cv2.imread('original/depth_filt_final.png')
  img2 = cv2.imread('merged/tresh3.png')
  result1 = cv2.add(img1,img2)
  cv2.imwrite('result1.png',result1)
  
  img2 = cv2.imread('merged/tresh5.png')
  result2 = cv2.add(img1,img2)
  cv2.imwrite('result2.png',result2)

def main():
    red, green, blue, depth = initgame()
    filtering(red, green, blue, depth)
    merging()
if __name__ == "__main__":
    main()
