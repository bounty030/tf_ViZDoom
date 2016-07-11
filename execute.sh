 #!/bin/bash    

#Parameter:
#1: Amount of stacked images
#2: Amount of frames until until a new action is predicted
#3: Amount of actions until epsioln is annealed to the final value (likely 5%)
#4: Sets if the depth buffer will be added to the input image

#stack 1
python main_dq.py 1 1 700000 1
#python main_dq.py 1 1 1000000 0
#python main_dq.py 1 3 1000000 1
#python main_dq.py 1 3 1000000 0
#stack 2
#python main_dq.py 2 1 1000000 1
#python main_dq.py 2 1 1000000 0
#python main_dq.py 2 3 1000000 1
#python main_dq.py 2 3 1000000 0
#stack 4
python main_dq.py 4 1 700000 1
#python main_dq.py 4 1 1000000 0
#python main_dq.py 4 3 1000000 1
python main_dq.py 4 3 700000 0
