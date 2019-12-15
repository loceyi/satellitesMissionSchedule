import pandas as pd
from interval import Interval
#关于action：若接收任务，action=1,拒绝任务，action=0
import globalVariableLocalWorker1
import RemainingTimeTotalModule
#self.state = np.array([Storage,TaskNumber,label])

def get_env_feedback(S, A):
    done=0
    # satStateTable=globalVariableLocal.get_value_satState()
    globalVariableLocalWorker1.taskListMove(S[1]) #更新完global值后要取出来
    taskList=globalVariableLocalWorker1.get_value_taskList()
    TaskTotal=globalVariableLocalWorker1.get_value_TaskTotal() #返回整个task字典变量
    # This is how agent will interact with the environment
    Tasknum = S[1]
    TaskRequirement=globalVariableLocalWorker1.get_value_Task(str(Tasknum)).copy()
    # S[2]=taskList[0]
    # RemainingTime = S[1]

    RemainingTimeTotal=RemainingTimeTotalModule.get_value_RemainingTimeTotal()
    RemainingTime = RemainingTimeTotal[S[2]].copy()

    # RemainingTime=RemainingTimeTotal[S[3]].copy() #因为取出来的是列表，只想复制它的值
    # print('S-label',S[3])
    # print('RemainingTime',RemainingTime)
    # print('Tasknum',Tasknum,'Action',A)
    # print('Task[str(Tasknum)][0] ',Task[str(Tasknum)][0])
    # print('Task[str(Tasknum)][1]',Task[str(Tasknum)][1])
    for i in range(0, len(RemainingTime)):

        if (TaskRequirement[0] in RemainingTime[i]) and (TaskRequirement[1] in RemainingTime[i]):


            NumTW = i

            break

    if A == 1: #Accept=1

        R = float(TaskRequirement[3])

        S[0] = S[0] - TaskRequirement[2]
        # 更新可用时间窗口
        # a=S[1]
        NewTW_1 = Interval(RemainingTime[NumTW].lower_bound, TaskRequirement[0], closed=True)
        NewTW_2 = Interval(TaskRequirement[1], RemainingTime[NumTW].upper_bound, closed=True)
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

        #更新下一个任务分配，如果下一个任务有冲突就跳到再下一个任务,一直验证到不冲突的任务再把任务给出去

        for i in range(0,len(taskList)):

            if taskList[0] == 0:

                S[1] = taskList[0]
                done=1

                break

            else:

                Counter=0

                for j in range(0, len(RemainingTime)):

                    if (TaskTotal[str(taskList[0])][0] in RemainingTime[j]) and\
                            (TaskTotal[str(taskList[0])][1] in RemainingTime[j]):

                        Counter += 1



                if S[0] < TaskTotal[str(taskList[0])][2] or Counter == 0:

                    # taskList.pop(0)  # 删除第一个元素
                    globalVariableLocalWorker1.taskListPop()
                    taskList = globalVariableLocalWorker1.get_value_taskList()
                    S[1] = taskList[0]



                else:

                    S[1] = taskList[0]

                    break

        # 判断此时的状态是否是之前的episode遍历过的
        #为什么需要判读：qtable中是为了对应状态的更新，这里是为了取出对应的timewindow
        diff = 0
        # 判断是否出现过同样的timewindow
        # print('RemainingTimeTotalBefore',RemainingTimeTotal)
        # print(Tasknum,A)

        for i in range(0, len(RemainingTimeTotal)):

            diff_TW = 0
            RemainingTime_i = RemainingTimeTotal[i].copy()
            CurrentStateRemaingingTime = RemainingTime.copy()
            CRT = len(CurrentStateRemaingingTime)
            RT = len(RemainingTime_i)

            if CRT != RT:

                diff_TW += 1


            else:
                # 由于窗口时间是被分成了几段interval存储，所以也要遍历
                for i_1 in range(0, CRT):

                    CurrentWindow = CurrentStateRemaingingTime[i_1]
                    ExisintWindow = RemainingTime_i[i_1]

                    if CurrentWindow.lower_bound != ExisintWindow.lower_bound:
                        diff_TW += 1

                        break

                    elif CurrentWindow.upper_bound != ExisintWindow.upper_bound:

                        diff_TW += 1

                        break

                    else:

                        pass
                # 判断若窗口全都一样，看看其它状态量是否相同
            if diff_TW == 0:

                S[2] = i #与RemainTimetotal中第i个时间窗口相同


            else:

                diff += 1

        if diff == len(RemainingTimeTotal):

            # new = pd.DataFrame({'Accept': 0,
            #                     'Reject': 0,
            #                     'Storage': S[0],
            #                     'IncomingTask': S[2]},
            #                    index=[0])
            #
            # q_table = q_table.append(new, ignore_index=True)
            # RemainingTimeTotal.append(RemainingTime)
            S[2]=len(RemainingTimeTotal)
            # globalVariableLocal.addNewState(S[0], S[1], S[2])
            RemainingTimeTotalModule.updateRemainTimeTotal(RemainingTime)


        else:

            pass


    else:

        R = float(0.01)

        # S[2] = taskList[0]
        # 更新下一个任务分配，如果下一个任务有冲突就跳到再下一个任务,一直验证到不冲突的任务再把任务给出去
        for i in range(0,len(taskList)):

            if taskList[0] == 0:

                S[1] = taskList[0]
                done = 1

                break

            else:

                Counter=0

                for j in range(0, len(RemainingTime)):

                    if (TaskTotal[str(taskList[0])][0] in RemainingTime[j]) and\
                            (TaskTotal[str(taskList[0])][1] in RemainingTime[j]):

                        Counter += 1



                if S[0] < TaskTotal[str(taskList[0])][2] or Counter == 0:

                    # taskList.pop(0)  # 删除第一个元素
                    #
                    # S[1] = taskList[0]

                    globalVariableLocalWorker1.taskListPop()
                    taskList = globalVariableLocalWorker1.get_value_taskList()
                    S[1] = taskList[0]




                else:

                    S[1] = taskList[0]

                    break

        # 判断此时的状态是否是之前的episode遍历过的
        diff = 0
        # 判断是否出现过同样的timewindow
        # print(RemainingTimeTotal)
        # print(Tasknum, A)
        for i in range(0, len(RemainingTimeTotal)):
            diff_TW = 0
            RemainTimeIndex = i
            RemainingTime_i = RemainingTimeTotal[RemainTimeIndex].copy()

            CurrentStateRemaingingTime = RemainingTime.copy()
            CRT = len(CurrentStateRemaingingTime)
            RT = len(RemainingTime_i)

            if CRT != RT:

                diff_TW += 1


            else:
                # 由于窗口时间是被分成了几段interval存储，所以也要遍历
                for i_1 in range(0, CRT):

                    CurrentWindow = CurrentStateRemaingingTime[i_1]
                    ExisintWindow = RemainingTime_i[i_1]

                    if CurrentWindow.lower_bound != ExisintWindow.lower_bound:
                        diff_TW += 1

                        break

                    elif CurrentWindow.upper_bound != ExisintWindow.upper_bound:

                        diff_TW += 1

                        break

                    else:

                        pass
                # 判断若窗口全都一样，看看其它状态量是否相同
            if diff_TW == 0:

                S[2] = i

            else:

                diff += 1

        if diff == len(RemainingTimeTotal):

            # new = pd.DataFrame({'Accept': 0,
            #                     'Reject': 0,
            #                     'Storage': S[0],
            #                     'IncomingTask': S[2]},
            #                    index=[0])
            #
            # q_table = q_table.append(new, ignore_index=True)
            # RemainingTimeTotal.append(RemainingTime)
            # S[3] = q_table.shape[0] - 1
            # S[1] = S[3]

            S[2]=len(RemainingTimeTotal)
            # globalVariableLocal.addNewState(S[0], S[1], S[2])
            RemainingTimeTotalModule.updateRemainTimeTotal(RemainingTime)

        else:

            pass

    return S,R,done

