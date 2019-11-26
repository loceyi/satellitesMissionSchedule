import test2
import test
test.add([1,2,3])

d=test.get_value() #必须要通过get_value才能把值传出来。
test.add([2,3,4])
# d=test.get_value()
print(d)
#python函数return只返回值，无论是数还是数组都不返回地址，如上所示，如果python返回值是地址，
#则d获取的就是数组a的地址，在运行test.add([2,3,4])后d的内容应该随a而改变，结果并没有。