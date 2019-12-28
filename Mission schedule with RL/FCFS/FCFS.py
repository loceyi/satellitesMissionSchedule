#First come first serve.
import numpy as np
import myEnvLocal as myEnv
from stepFunction import get_env_feedback
from interval import Interval
Task = {
            '3': [64, 64, 1, 100, -0.000476265673515597],
            '7': [68, 68, 1, 40, -0.000506125053969096],
            '2': [68, 68, 1, 100, 0.000452382421992215],
            '14':[118,118,1, 10,  -0.00134238388121341],
            '16': [119, 119, 1, 10, -0.000619947849442742],
            '1': [124, 124, 1, 100, 0],
            '13':[132,132,1,10,0.000831697803189602],
            '8': [151, 151, 1, 40, -0.000189067976374232],
            '12':[210,210,1,40, 0.000686616104577854],
            '6': [222, 222, 1, 100, -0.00034561912981283],
            '9': [266, 266, 1, 40, -0.000627053588052687],
            '15': [281, 281, 1, 10, -0.000249234646338898],
            '4': [313, 313, 1, 100, 0.000346407017990367],
            '10': [424, 424, 1, 40, 0.00014618739102126],
            '17': [441, 441, 1, 10, -0.000132150261068142],
            '5': [493, 493, 1, 100, -0.000143827287224467],
            '11': [493, 493, 1, 40, -0.00006707423124954]
            }

taskList=[3,7,2,14,16,1,13,8,12,6,9,15,4,10,17,5,11]
RemainingTime = [Interval(0, 500, closed=True)]
Storage = 10
TaskNumber = 3
label = 0
angle=0
S = np.array([Storage,TaskNumber,angle])

TotalReward=0
for i in range(0,17):

    S[1] = taskList[i]
    task=Task[str(taskList[i])]

    S,A,R,RemainingTime = get_env_feedback(S,task,RemainingTime)


    TotalReward=TotalReward+R

    print('task',S[1],'Action',A)
print('Total Reward', TotalReward)

# [[3, 1], [14, 1], [8, 1], [12, 1], [9, 1], [4, 1], [10, 1], [5, 1]]