#考虑一个任务有重复窗口的问题
#初步假设如果任务因为冲突没有被安排，再去考虑安排后续窗口，
#每一个规划总时间段内不能有同一个目标的不同窗口.
#用stk计算时间窗口

# 2018年7月24日22时19分52秒	2018年7月24日22时20分37秒	45.501	YuannanProvince
# 2018年7月24日22时22分32秒	2018年7月24日22时23分58秒	85.691	HenanProvince
# 2018年7月24日22时22分54秒	2018年7月24日22时24分11秒	76.985	Anhui1
# 2018年7月24日22时23分51秒	2018年7月24日22时25分16秒	85.356	ShandongProvince
# 2018年7月24日22时24分02秒	2018年7月24日22时25分27秒	85.599	Shouguang
# StartTime 737265.930462963	737265.932314815
# 737265.932569444	737265.933229167	737265.933356482
# EndTime   737265.930983796	737265.933310185
# 737265.933460648	737265.934212963	737265.934340278
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from interval import Interval

# State S[Remaining data storage ability, Remaining time,Incoming Task number]
# Task [startTime,endTime,engergyCost,reward]


#20个任务的情况
# Task = {'1': [737265.930462963, 737265.930983796, 1, 2],
#         '2': [737265.932314815, 737265.933310185, 1, 2],
#         '3': [737265.932569444, 737265.933460648, 1, 2],
#         '4': [737265.933229167, 737265.934212963, 1, 2],
#         '5': [737265.933356482, 737265.934340278, 1, 2],
#         '6': [737265.940462963, 737265.940983796, 1, 2],
#         '7': [737265.942314815, 737265.943310185, 1, 2],
#         '8': [737265.942569444, 737265.943460648, 1, 2],
#         '9': [737265.943229167, 737265.944212963, 1, 2],
#         '10':[737265.943356482, 737265.944340278, 1, 2],
#         '11': [737265.950462963, 737265.950983796, 1, 2],
#         '12': [737265.952314815, 737265.953310185, 1, 2],
#         '13': [737265.952569444, 737265.953460648, 1, 2],
#         '14': [737265.953229167, 737265.954212963, 1, 2],
#         '15': [737265.953356482, 737265.954340278, 1, 2],
#         '16': [737265.960462963, 737265.960983796, 1, 2],
#         '17': [737265.962314815, 737265.963310185, 1, 2],
#         '18': [737265.962569444, 737265.963460648, 1, 2],
#         '19': [737265.963229167, 737265.964212963, 1, 2],
#         '20':[737265.963356482, 737265.964340278, 1, 2]
#         }


#taskGroup1
Task = {'1': [737265.930462963, 737265.930983796, 1, 2],
        '2': [737265.932314815, 737265.933310185, 1, 2],
        '3': [737265.932569444, 737265.933460648, 1, 2],
        '4': [737265.933229167, 737265.934212963, 1, 2],
        '5': [737265.933356482, 737265.934340278, 1, 2],
        }
#
# taskGroup2 = {'1b': [737265.930462963, 737265.930983796, 1, 2],
#         '11': [737265.932314815, 737265.933310185, 1, 2],
#         '13': [737265.932569444, 737265.933460648, 1, 2],
#         '14': [737265.933229167, 737265.934212963, 1, 2],
#         '15': [737265.933356482, 737265.934340278, 1, 2],
#         '16': [737265.940462963, 737265.940983796, 1, 2],
#         '17': [737265.942314815, 737265.943310185, 1, 2],
#         '18': [737265.942569444, 737265.943460648, 1, 2],
#         '19': [737265.943229167, 737265.944212963, 1, 2],
#         '20': [737265.943356482, 737265.944340278, 1, 2]
#         }



Tasklist_Initial = [1,2,3,4,5,0]
RemainingTime_Initial = [Interval(Task['1'][0], Task['5'][1], closed=True)]
RemainingTimeTotal = [[Interval(Task['1'][0], Task['5'][1], closed=True)]]
Storage = 5
TaskNumber = 1
label = 0
S = [Storage, RemainingTime_Initial, TaskNumber, label]
N_STATES = 1  # 1维世界的宽度
ACTIONS = ['Accept', 'Reject', 'Storage', 'IncomingTask']  # 探索者的可用动作
EPSILON = 0.8  # 贪婪度 greedy
ALPHA = 0.1  # 学习率
GAMMA = 0.9  # 奖励递减值
MAX_EPISODES = 1000 # 最大回合数
FRESH_TIME = 0.3  # 移动间隔时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table 全 0 初始
        columns=actions,  # columns 对应的是行为名称
    )

    table.loc[0, 'Storage'] = 5
    table.loc[0, 'IncomingTask'] = 1

    return table


# q_table:
"""
   left  right
0   0.0    0.0
1   0.0    0.0
2   0.0    0.0
3   0.0    0.0
4   0.0    0.0
5   0.0    0.0
"""


# 在某个 state 地点, 选择行为
def choose_action(S, q_table):
    state_actions = q_table.iloc[S[3], 0:2]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS[0:2])
    else:
        action_name = state_actions.idxmax()  # 贪婪模式
    return action_name

def choose_action_greedy(S, q_table):

    state_actions = q_table.iloc[S[3], 0:2]  # 选出这个 state 的所有 action 值
    action_name = state_actions.idxmax()  # 贪婪模式

    return action_name








def get_env_feedback(S, A, taskList, q_table, RemainingTimeTotal):
    # This is how agent will interact with the environment
    taskList.remove(S[2])  # 确保一个episode里面只会遇到某一个任务一次
    Tasknum = S[2]
    RemainingTime = S[1]
    Counter = 0
    for i in range(0, len(RemainingTime)):

        if (Task[str(Tasknum)][0] in RemainingTime[i]) and (Task[str(Tasknum)][1] in RemainingTime[i]):
            Counter += 1

            NumTW = i

            break

    if S[0] < Task[str(Tasknum)][2] or Counter == 0:

        if A == 'Accept':

            R = -1

            S[2] = taskList[0]

            # 判断此时的状态是否是之前的episode遍历过的
            diff = 0
            # 判断是否出现过同样的timewindow

            for i in range(0, q_table.shape[0]):
                diff_TW = 0
                RemainTimeIndex = i
                RemainingTime = RemainingTimeTotal[RemainTimeIndex]
                CurrentStateRemaingingTime = S[1]
                CRT = len(CurrentStateRemaingingTime)
                RT = len(RemainingTime)

                if CRT != RT:

                    diff_TW += 1


                else:
                    # 由于窗口时间是被分成了几段interval存储，所以也要遍历
                    for i_1 in range(0, CRT):

                        CurrentWindow = CurrentStateRemaingingTime[i_1]
                        ExisintWindow = RemainingTime[i_1]

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

                    if S[0] != q_table.loc[i, 'Storage']:
                        diff += 1



                    else:

                        if S[2] != q_table.loc[i, 'IncomingTask']:
                            diff += 1


                        else:

                            SameRecord = i
                            S[3] = SameRecord
                            break
                else:

                    diff += 1

            if diff == q_table.shape[0] - 1:

                new = pd.DataFrame({'Accept': 0,
                                    'Reject': 0,
                                    'Storage': S[0],
                                    'IncomingTask': S[2]},
                                   index=[0])

                q_table = q_table.append(new, ignore_index=True)
                RemainingTimeTotal.append(S[1])
                S[3] = q_table.shape[0] - 1


            else:

                pass





        else:

            R = 0

            S[2] = taskList[0]

            # 判断此时的状态是否是之前的episode遍历过的
            diff = 0
            # 判断是否出现过同样的timewindow

            for i in range(0, q_table.shape[0]):
                diff_TW = 0
                RemainTimeIndex = i
                RemainingTime = RemainingTimeTotal[RemainTimeIndex]
                CurrentStateRemaingingTime = S[1]
                CRT = len(CurrentStateRemaingingTime)
                RT = len(RemainingTime)

                if CRT != RT:

                    diff_TW += 1


                else:
                    # 由于窗口时间是被分成了几段interval存储，所以也要遍历
                    for i_1 in range(0, CRT):

                        CurrentWindow = CurrentStateRemaingingTime[i_1]
                        ExisintWindow = RemainingTime[i_1]

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

                    if S[0] != q_table.loc[i, 'Storage']:
                        diff += 1



                    else:

                        if S[2] != q_table.loc[i, 'IncomingTask']:
                            diff += 1


                        else:

                            SameRecord = i
                            S[3] = SameRecord

                            break

                else:

                    diff += 1

            if diff == q_table.shape[0]:

                new = pd.DataFrame({'Accept': 0,
                                    'Reject': 0,
                                    'Storage': S[0],
                                    'IncomingTask': S[2]},
                                   index=[0])

                q_table = q_table.append(new, ignore_index=True)
                RemainingTimeTotal.append(S[1])
                S[3] = q_table.shape[0] - 1


            else:

                pass


    else:

        if A == 'Accept':

            R = Task[str(Tasknum)][3]

            S[0] = S[0] - Task[str(Tasknum)][2]
            # 更新可用时间窗口
            # a=S[1]
            RemainingTime = S[1].copy()
            NewTW_1 = Interval(RemainingTime[NumTW].lower_bound, Task[str(Tasknum)][0], closed=True)
            NewTW_2 = Interval(Task[str(Tasknum)][1], RemainingTime[NumTW].upper_bound, closed=True)
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

            S[1] = RemainingTime

            S[2] = taskList[0]

            # 判断此时的状态是否是之前的episode遍历过的
            diff = 0
            # 判断是否出现过同样的timewindow

            for i in range(0, q_table.shape[0]):
                diff_TW = 0
                RemainTimeIndex = i
                RemainingTime = RemainingTimeTotal[RemainTimeIndex]
                CurrentStateRemaingingTime = S[1]
                CRT = len(CurrentStateRemaingingTime)
                RT = len(RemainingTime)

                if CRT != RT:

                    diff_TW += 1


                else:
                    # 由于窗口时间是被分成了几段interval存储，所以也要遍历
                    for i_1 in range(0, CRT):

                        CurrentWindow = CurrentStateRemaingingTime[i_1]
                        ExisintWindow = RemainingTime[i_1]

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

                    if S[0] != q_table.loc[i, 'Storage']:
                        diff += 1



                    else:

                        if S[2] != q_table.loc[i, 'IncomingTask']:
                            diff += 1


                        else:

                            SameRecord = i
                            S[3] = SameRecord

                            break

                else:

                    diff += 1

            if diff == q_table.shape[0]:

                new = pd.DataFrame({'Accept': 0,
                                    'Reject': 0,
                                    'Storage': S[0],
                                    'IncomingTask': S[2]},
                                   index=[0])

                q_table = q_table.append(new, ignore_index=True)
                RemainingTimeTotal.append(S[1])
                S[3] = q_table.shape[0] - 1


            else:

                pass


        else:

            R = 0

            S[2] = taskList[0]

            # 判断此时的状态是否是之前的episode遍历过的
            diff = 0
            # 判断是否出现过同样的timewindow

            for i in range(0, q_table.shape[0]):
                diff_TW = 0
                RemainTimeIndex = i
                RemainingTime = RemainingTimeTotal[RemainTimeIndex]
                CurrentStateRemaingingTime = S[1]
                CRT = len(CurrentStateRemaingingTime)
                RT = len(RemainingTime)

                if CRT != RT:

                    diff_TW += 1


                else:
                    # 由于窗口时间是被分成了几段interval存储，所以也要遍历
                    for i_1 in range(0, CRT):

                        CurrentWindow = CurrentStateRemaingingTime[i_1]
                        ExisintWindow = RemainingTime[i_1]

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

                    if S[0] != q_table.loc[i, 'Storage']:
                        diff += 1



                    else:

                        if S[2] != q_table.loc[i, 'IncomingTask']:
                            diff += 1


                        else:

                            SameRecord = i
                            S[3] = SameRecord

                            break

                else:

                    diff += 1

            if diff == q_table.shape[0]:

                new = pd.DataFrame({'Accept': 0,
                                    'Reject': 0,
                                    'Storage': S[0],
                                    'IncomingTask': S[2]},
                                   index=[0])

                q_table = q_table.append(new, ignore_index=True)
                RemainingTimeTotal.append(S[1])
                S[3] = q_table.shape[0] - 1


            else:

                pass

    return S, R, q_table, RemainingTimeTotal


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S[2] == 3:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl(RemainingTimeTotal,RemainingTime_Initial,Tasklist_Initial,Storage):
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    episodeCounter = 0
    rewardCounter=[]
    for episode in range(MAX_EPISODES):

        episodeCounter += 1
        # 回合
        RemainingTime = RemainingTime_Initial.copy()
        TaskNumber = 1
        label = 0
        S = [Storage, RemainingTime, TaskNumber, label]
        Tasklist = Tasklist_Initial.copy()
        rEachEpisode = 0
        is_terminated = False  # 是否回合结束
        # update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)  # 选行为
            S_old = S.copy()#注意，数组传入函数中自身会被改变，所以要用copy
            S_, R, q_table, RemainingTimeTotal = get_env_feedback(S, A, Tasklist, q_table,
                                                                  RemainingTimeTotal)  # 实施行为并得到环境的反馈

            q_predict = q_table.loc[S_old[3], A]  # 估算的(状态-行为)值
            # q_predict = q_table.loc[S[3], A]
            if S_[2] != 0:
                q_target = R + GAMMA * q_table.iloc[S_[3], 0:2].max()  # 实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True  # terminate this episode

            q_table.loc[S_old[3], A] += ALPHA * (q_target - q_predict)  # q_table 更新

            #q_table.loc[S[3], A] += ALPHA * (q_target - q_predict)
            S = S_  # 探索者移动到下一个 state
            rEachEpisode += R
            # update_env(S, episode, step_counter+1)  # 环境更新

        rewardCounter.append(rEachEpisode)
        # step_counter += 1
    return q_table,RemainingTimeTotal,rewardCounter



def getSolution(q_table,RemainingTimeTotal,RemainingTime_Initial,Tasklist_Initial, Storage ):
    action_space=[]
    TaskNumber = 1
    label = 0
    RemainingTime=RemainingTime_Initial.copy()
    RemainingTimeTotal_gets = RemainingTimeTotal.copy()
    S = [Storage, RemainingTime, TaskNumber, label]
    Tasklist=Tasklist_Initial.copy()
    Reward=0

    is_terminated = False  # 是否回合结束
    # update_env(S, episode, step_counter)    # 环境更新
    while not is_terminated:

        A = choose_action_greedy(S, q_table)  # 选行为
        action_space.append(A)
        S_, R, q_table, RemainingTimeTotal_gets = get_env_feedback(S, A, Tasklist, q_table,
                                                                   RemainingTimeTotal_gets)  # 实施行为并得到环境的反馈
        if S_[2] != 0:

            pass

        else:
            is_terminated = True  # terminate this episode


        S = S_  # 探索者移动到下一个 state
        Reward+=R

    return action_space,Reward


if __name__ == "__main__":
    q_table,RemainingTimeTotal,reward_counter = rl(RemainingTimeTotal,
                                                   RemainingTime_Initial,
                                                   Tasklist_Initial,Storage)

    action_space,Reward=getSolution(q_table,RemainingTimeTotal,
                                    RemainingTime_Initial,
                                    Tasklist_Initial, Storage)
    print('action_space',action_space,'Reward',Reward)


    plt.plot(reward_counter)
    plt.show()
    # print('\r\nQ-table:\n')
    # print(q_table)
    # print('Time')
    # print(RemainingTimeTotal)

