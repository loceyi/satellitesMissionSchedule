a=[1,2,3]
print(a)

#列表传入数组会被改变
def aad(b):

    b[0]=2

    return b
b_=aad(a)
print(a)

t=1
c=1
S=[t,c]
S[0]=2
print(t,c,S)

k=[2,3,4,5]

c=k #c与k指向同一列表
d=k
k[0]=1 #修改指向列表的值
k=[1,2,3,4]#重新给k指派列表

c[1]=20

print(d)



e=1
l=e

e=2

print(l,e)