#!/usr/bin/python
from __future__ import print_function
from time import sleep
from random import choice
from initGame import initgame
from initNetwork import *


def trainNetwork(actions, game, s, readout, h_fc1, sess):
    episodes = 20

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup. 
    sleep_time = 10.028

    for i in range(episodes):
        print("Episode #" + str(i+1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        while not game.is_episode_finished():

        # Gets the state
            s = game.get_state()

        # Makes a random action and get remember reward.
            r = game.make_action(choice(actions))
            print(choice(actions))

        # Prints state's game variables. Printing the image is quite pointless.
            print("State #" + str(s.number))
            print("Game variables:", s.game_variables[0])
            print("Reward:", r)
            print("=====================")

            if sleep_time>0:
                sleep(sleep_time)

    # Check how the episode went.
        print("Episode finished.")
        print("total reward:", game.get_total_reward())
        print("************************")
        
    game.close()
    
def main():
    actions, num_actions, game = initgame()
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork(num_actions)
    trainNetwork(actions, game, s, readout, h_fc1, sess)
    
if __name__ == "__main__":
    main()