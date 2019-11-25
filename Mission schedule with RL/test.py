
import numpy as np
import pandas as pd



def init():
    global a
    a=[]


def add(b):

    global a

    a=b


def get_value():
    global a


    return a

if __name__ == "__main__":

    init()

    get_value()
    print(a)
