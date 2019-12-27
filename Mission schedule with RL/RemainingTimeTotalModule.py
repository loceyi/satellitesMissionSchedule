from interval import Interval
import copy

def initRemainingTimeTotal():

    global RemainingTimeTotal

    Task = {'18': [0, 0, 1, 10, 0],
            '3': [64, 64, 1, 100, -0.000476265673515597],
            '7': [68, 68, 1, 40, -0.000506125053969096],
            '2': [68, 68, 1, 100, 0.000452382421992215],
            '14': [118, 118, 1, 10, -0.00134238388121341],
            '16': [119, 119, 1, 10, -0.000619947849442742],
            '1': [124, 124, 1, 100, 0],
            '13': [132, 132, 1, 10, 0.000831697803189602],
            '8': [151, 151, 1, 40, -0.000189067976374232],
            '12': [210, 210, 1, 40, 0.000686616104577854],
            '6': [222, 222, 1, 100, -0.00034561912981283],
            '9': [266, 266, 1, 40, -0.000627053588052687],
            '15': [281, 281, 1, 10, -0.000249234646338898],
            '4': [313, 313, 1, 100, 0.000346407017990367],
            '10': [424, 424, 1, 40, 0.00014618739102126],
            '17': [441, 441, 1, 10, -0.000132150261068142],
            '5': [493, 493, 1, 100, -0.000143827287224467],
            '11': [493, 493, 1, 40, -0.00006707423124954]

            }

    RemainingTimeTotal = [[Interval(-100, 500, closed=True)]]
    #初始时间要早一些，因为要留一定时间给卫星机动到第一个任务

def updateRemainTimeTotal(RemainingTime):

    global RemainingTimeTotal

    RemainingTimeTotal.append(RemainingTime)

def get_value_RemainingTime(label):

    global RemainingTimeTotal

    RemainingTime=RemainingTimeTotal[label].copy()
    # storage=satStateTable.loc[label, 'storage']
    # nextTask=satStateTable.loc[label, 'nextTask']


    return RemainingTime


def get_value_RemainingTimeTotal():

    global RemainingTimeTotal

    return copy.deepcopy(RemainingTimeTotal)
