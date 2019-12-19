from interval import Interval
import copy

def initRemainingTimeTotal():

    global RemainingTimeTotal

    Task = {'3': [65, 65, 1, 100, -7.45724275766466],
            '7': [68, 68, 1, 40, -7.89969417530144],
            '2': [69, 69, 1, 100, 13.6399007451127],
            '16': [118, 118, 1, 10, -10.3682107759176],
            '1': [124, 124, 1, 100, 11.6271525374774],
            '8': [150, 150, 1, 40, -1.84496651971562],
            '6': [218, 218, 1, 100, -4.59939101003146],
            '9': [258, 258, 1, 40, -10.9580854925028],
            '15': [272, 272, 1, 10, -2.47942644716451],
            '4': [300, 300, 1, 100, 12.1725862458616],
            '10': [392, 392, 1, 40, 7.43455315035822],
            '17': [406, 406, 1, 10, 2.10370861521673],
            '5': [446, 446, 1, 100, 1.90096462779665],
            '11': [446, 446, 1, 40, 3.09407087944975],
            '18': [473, 473, 1, 10, -6.84203655806705]
            }

    RemainingTimeTotal = [[Interval(0, 500, closed=True)]]
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
