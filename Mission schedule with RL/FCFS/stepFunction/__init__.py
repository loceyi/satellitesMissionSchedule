
from interval import Interval

def get_env_feedback(S,task,RemainingTime):
    a_omega=0.2 #平均角加速度°/s^2
    maxOmega=2 #最大角速度 °/s
    ms_b=15 #秒s
    ms_e=5 #秒s

    rollAngle=S[2]





    '''
    根据当前卫星的roll angle，以及当前来临任务的roll angle,计算需要的
    机动时间，将时间添加进来临Task的时间窗口内
    '''

    taskRollAngle = task[4] #roll angle of incoming task
    deltaRollAngle=abs(taskRollAngle-rollAngle)#需要机动的角度
    if deltaRollAngle <= pow(maxOmega,2)/a_omega:

        attitudeManeuverTimeSeconds= 2*pow(deltaRollAngle/a_omega,0.5)+ms_b+ms_e

    else:

        attitudeManeuverTimeSeconds=(a_omega*deltaRollAngle-pow(maxOmega,2))/(a_omega*maxOmega)+ms_b+ms_e

    # attitudeManeuverTimeJulian= attitudeManeuverTimeSeconds/86400.0
    # print(attitudeManeuverTimeJulian)
    #修改incoming task的起始时间
    task[0] = task[0]-attitudeManeuverTimeSeconds




    #判断来临任务是否与卫星状态冲突

    Counter = 0
    # 计算预分配任务是否拥有足够的机动时间，如果没有，则不分配


    for j in range(0, len(RemainingTime)):
        # print('RemainingTime',RemainingTime)
        # print('n',task[0] - attitudeManeuverTimeSeconds)
        if (task[0]   in RemainingTime[j]) and \
                (task[1]  in RemainingTime[j]):
            Counter += 1

        else:

            pass

    if S[0] < task[2] or Counter == 0:

        A=0



    else:

        A=1



    if A == 1: #Accept=1

        for i in range(0, len(RemainingTime)):

            if (task[0] in RemainingTime[i]) and (task[1] in RemainingTime[i]):
                NumTW = i
                # print(NumTW)

                break

        R = float(task[3])
        S[2]= taskRollAngle #更新卫星姿态
        S[0] = S[0] - task[2]
        # 更新可用时间窗口
        # a=S[1]
        NewTW_1 = Interval(RemainingTime[NumTW].lower_bound, task[0], closed=True)
        NewTW_2 = Interval(task[1], RemainingTime[NumTW].upper_bound, closed=True)
        if NewTW_1.upper_bound - NewTW_1.lower_bound == 0:

            if NewTW_2.upper_bound - NewTW_2.lower_bound == 0:

                RemainingTime.pop(NumTW)


            else:

                RemainingTime.insert(NumTW + 1, NewTW_2)
                RemainingTime.pop(NumTW)

        else:

            if NewTW_2.upper_bound - NewTW_2.lower_bound == 0:

                RemainingTime.insert(NumTW, NewTW_1)
                RemainingTime.pop(NumTW + 1)

            else:

                RemainingTime.insert(NumTW, NewTW_1)
                RemainingTime.insert(NumTW + 2, NewTW_2)
                RemainingTime.pop(NumTW + 1)


    else:

        R = 0


    return S,A,R,RemainingTime

