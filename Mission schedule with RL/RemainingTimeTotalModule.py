from interval import Interval
import copy

def initRemainingTimeTotal():

    global RemainingTimeTotal

    Task = {'1': [737265.930462963, 737265.930983796, 1, 2],
            '2': [737265.932314815, 737265.933310185, 1, 2],
            '3': [737265.932569444, 737265.933460648, 1, 2],
            '4': [737265.933229167, 737265.934212963, 1, 2],
            '5': [737265.933356482, 737265.934340278, 1, 2]
            }

    RemainingTimeTotal = [[Interval(Task['1'][0], Task['5'][1], closed=True)]]


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
