#satStateTable被定义为全局变量方便各个部分调用
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#satStateTable [label storage timeWindow nextTask]


def _init():
    global satStateTable
    satStateTable = pd.DataFrame(
        np.zeros((1, 4)),
        columns=['label','storage','timeWindow','nextTask'])

# def set_value(name, value):
#
#     _global_dict[name] = value


def addNewState(label,storage,timeWindow,nextTask):
    global satStateTable
    new = pd.DataFrame({'label': label,
                        'storage': storage,
                        'timeWindow': timeWindow,
                        'nextTask': nextTask},
                       index=[0])#设置行初始index

    satStateTable = satStateTable.append(new, ignore_index=True)


def get_value(label):
    global satStateTable

    storage=satStateTable.loc[label, 'storage']
    timeWindow=satStateTable.loc[label, 'timeWindow']
    nextTask=satStateTable.loc[label, 'nextTask']

    return storage,timeWindow,nextTask
