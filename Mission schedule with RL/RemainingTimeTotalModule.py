from interval import Interval
import copy

def initRemainingTimeTotal():

    global RemainingTimeTotal



    RemainingTimeTotal={'0.5':[Interval(0, 500, closed=True)]}

    # RemainingTimeTotal = [[Interval(0, 500, closed=True)]]
    #初始时间要早一些，因为要留一定时间给卫星机动到第一个任务

def updateRemainTimeTotal(label,RemainingTime):

    global RemainingTimeTotal

    RemainingTimeTotal[str(label)]=RemainingTime
    # print('s',RemainingTimeTotal)
def get_value_RemainingTime(label):

    global RemainingTimeTotal

    RemainingTime=RemainingTimeTotal[str(label)].copy()
    # storage=satStateTable.loc[label, 'storage']
    # nextTask=satStateTable.loc[label, 'nextTask']


    return RemainingTime


def get_value_RemainingTimeTotal():

    global RemainingTimeTotal

    return copy.deepcopy(RemainingTimeTotal)
