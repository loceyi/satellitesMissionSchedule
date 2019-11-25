import test

def c(k):

    test.init()#通过调用函数来在test变量空间里面操作a

    test.add(k)

    # a=test.get_value() #必须要通过get_value才能把值传出来。
    #
    #
    # print(a)

if __name__ == '__main__':
    k=213
    c(k)