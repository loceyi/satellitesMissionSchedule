import test


test.init()#通过调用函数来在test变量空间里面操作a

test.add(5)

a=test.get_value() #必须要通过get_value才能把值传出来。


print(a)