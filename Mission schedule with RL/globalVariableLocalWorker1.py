#satStateTable被定义为全局变量方便各个部分调用
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from interval import Interval
import copy
#satStateTable [label storage timeWindow nextTask]

# def initsatState():
#
#     global satStateTable
#
#     satStateTable = pd.DataFrame(
#         np.zeros((1, 3)),
#         columns=['Storage','TaskNumber','label'])
#
#     satStateTable.loc[0, 'Storage'] = 5
#     satStateTable.loc[0, 'TaskNumber'] = 1
#     satStateTable.loc[0, 'label'] = 0 #状态label从零开始编号，代表不同的状态
#     # 最后的label确保了状态不会重叠编成一样的。

def initTasklist():
    # global satStateTable
    # global  Task
    # global RemainingTimeTotal
    global taskList
    # satStateTable = pd.DataFrame(
    #     np.zeros((1, 3)),
    #     columns=['Storage','TaskNumber','label'])
    taskList=[1,2,3,4,5,0]
    # Task[startTime, endTime, engergyCost, reward]


def initTask():

    global Task

    Task = {'1': [737265.930462963, 737265.930983796, 1, 2],
            '2': [737265.932314815, 737265.933310185, 1, 2],
            '3': [737265.932569444, 737265.933460648, 1, 2],
            '4': [737265.933229167, 737265.934212963, 1, 2],
            '5': [737265.933356482, 737265.934340278, 1, 2]
            }



# def initRemainingTimeTotal():
#
#     global RemainingTimeTotal
#     global Task
#
#     RemainingTimeTotal = [[Interval(Task['1'][0], Task['5'][1], closed=True)]]


# def updateRemainTimeTotal(RemainingTime):
#
#     global RemainingTimeTotal
#
#     RemainingTimeTotal.append(RemainingTime)





# def set_value(name, value):
#
#     _global_dict[name] = value


def addNewState(storage,nextTask,label):
    global satStateTable
    global Task



    # Tasklist_Initial = [1, 2, 3, 4, 5, 0]


    new = pd.DataFrame({'Storage':storage ,
                        'TaskNumber':nextTask ,
                        'label': label},
                       index=[0])#设置行初始index

    satStateTable = satStateTable.append(new, ignore_index=True)

def get_value_taskList():

    global taskList

    return taskList.copy()


def get_value_Task(number):

    #number为str类型数值
    global Task

    return Task[number].copy()

def get_value_TaskTotal():
    global Task

    return copy.deepcopy(Task)




# def get_value_RemainingTime(label):
#
#     global RemainingTimeTotal
#
#     RemainingTime=RemainingTimeTotal[label].copy()
#     # storage=satStateTable.loc[label, 'storage']
#     # nextTask=satStateTable.loc[label, 'nextTask']
#
#
#     return RemainingTime


# def get_value_RemainingTimeTotal():
#
#     global RemainingTimeTotal
#
#
#     # storage=satStateTable.loc[label, 'storage']
#     # nextTask=satStateTable.loc[label, 'nextTask']
#
#
#     return copy.deepcopy(RemainingTimeTotal)
    #为了防止把地址传出去误改了，确保所有改变值的操作都在本文件的变量空间中进行






def get_value_satState():

    global satStateTable

    return satStateTable






def taskListMove(number):

    global taskList

    taskList.remove(number)

def taskListPop():

    global taskList

    taskList.pop(0)




