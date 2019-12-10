import csv
import pandas as pd
import matplotlib.pyplot as plt
'''指定读取某一行'''
# with open('ThirtyTwoTargets.csv','r') as csvfile:
#     reader = csv.reader(csvfile)
#     column1 = [row[0]for row in reader]
#     print(column1)


# 下面是按照列属性读取的
Access_Start = pd.read_csv('ThirtyTwoTargets.csv',
                usecols=['Access Start (UTCG)'])
Access_Start_list=Access_Start.iloc[:,0].tolist()

Access_End = pd.read_csv('ThirtyTwoTargets.csv',
                usecols=['Access End (UTCG)'])
Access_End_list=Access_End.iloc[:,0].tolist()

for i in range(len(Access_Start_list)):

    x = [[Access_Start_list[i], Access_End_list[i]]] # 要连接的两个点的坐标
    y = [[i/10, i/10]]
    plt.plot(x[0], y[0], color='r')
    plt.scatter(x[0], y[0], s=5,color='b')
    plt.text(x[0][0], y[0][0], '% i' % (i+1), fontsize=10)
    # ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

plt.show()


Task_dict = {}




for i in range(len(Access_Start_list)):

    Task_dict.update({'%i' % i: [Access_Start_list[i],Access_End_list[i]
                                 ,1,2]})

# print(Task_dict)
# a=Access_Start.iloc[1,0]
# b=Access_End.iloc[1,0]



# d = pd.read_csv('D:\Data\新建文件夹\list3.2.csv', usecols=['case', 'roi', 'eq. diam.','x loc.','y loc.','slice no.'],
#                 nrows=10)
# 这是表示读取前10行