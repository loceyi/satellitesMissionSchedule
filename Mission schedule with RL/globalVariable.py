#satStateTable被定义为全局变量方便各个部分调用
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from interval import Interval
#satStateTable [label storage timeWindow nextTask]


def _init():
    global satStateTable
    global  Task
    global RemainingTimeTotal
    global taskList
    satStateTable = pd.DataFrame(
        np.zeros((1, 4)),
        columns=['label','storage','timeWindow','nextTask'])
    taskList=[1,2,3,4,5,0]
    # Task[startTime, endTime, engergyCost, reward]
    Task = {'1': [737265.930462963, 737265.930983796, 1, 2],
            '2': [737265.932314815, 737265.933310185, 1, 2],
            '3': [737265.932569444, 737265.933460648, 1, 2],
            '4': [737265.933229167, 737265.934212963, 1, 2],
            '5': [737265.933356482, 737265.934340278, 1, 2]
            }
    RemainingTimeTotal = [[Interval(Task['1'][0], Task['5'][1], closed=True)]]

# def set_value(name, value):
#
#     _global_dict[name] = value


def addNewState(label,storage,nextTask):
    global satStateTable
    global Task



    # Tasklist_Initial = [1, 2, 3, 4, 5, 0]


    new = pd.DataFrame({'label': label,
                        'storage': storage,
                        'nextTask': nextTask},
                       index=[0])#设置行初始index

    satStateTable = satStateTable.append(new, ignore_index=True)

def get_value_taskList():

    global taskList

    return taskList


def get_value_Task(number):

    #number为str类型数值
    global Task

    return Task[number]


def get_value_RemainingTime(label):

    global RemainingTimeTotal

    RemainingTime=RemainingTimeTotal[label].copy
    # storage=satStateTable.loc[label, 'storage']
    # nextTask=satStateTable.loc[label, 'nextTask']


    return RemainingTime

def taskListMove(number):

    global taskList

    taskList.remove(number)



