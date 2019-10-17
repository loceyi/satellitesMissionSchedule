import numpy as np
import pandas as pd
import time
from interval import Interval
#State S[Remaining data storage ability, Remaining time,Incoming Task number]

Task={'1':[1,3,2,2],'2':[2,5,3,3],'3':[6,8,2,2]}
Tasklist=[1,2,3]
RemainingTime=[Interval(1,8,closed=True)]
RemainingTimeTotal=[RemainingTime]
Tasklist=[2,3,2]
Storage=5
TaskNumber=1
label=0
S=[Storage, RemainingTime,TaskNumber,label]
N_STATES = 1  # 1维世界的宽度
ACTIONS = ['Accept', 'Reject','Storage','RemainTime','IncomingTask']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 10  # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
        columns=actions,    # columns 对应的是行为名称
    )

    table.loc[0,'Storage']=5
    table.loc[0, 'RemainTime']=0
    table.loc[0, 'IncomingTask']=1

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
    state_actions = q_table.iloc[S[3],0:2]  # 选出这个 state 的所有 action 值
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()    # 贪婪模式
    return action_name

def get_env_feedback(S, A,taskList,q_table):
    # This is how agent will interact with the environment
    taskList.remove(S[2]) #确保一个episode里面只会遇到某一个任务一次
    Tasknum=S[2]
    RemainingTime=S[1]

    if S[0] < Task[Tasknum][2] or \
            (Task[Tasknum][0] not in RemainingTime[0]) or (Task[Tasknum][1] not in RemainingTime[0]):

        if A=='Accept':

            R=-1

            S[2]=taskList[0]




        else:

            R=0

            S[2] = taskList[0]


    else:

        if A=='Accept':

            R=1

            S[0]=S[0]-Task[Tasknum][2]
            #更新可用时间窗口
            for i in range(0,len(RemainingTime)):


                RemainingTime[1].lower_bound = RemainingTime[0].lower_bound
                RemainingTime[0].lower_bound=Task[Tasknum][1]
                RemainingTime[1].upper_bound=Task[Tasknum][0]

            S[2] = taskList[0]

            # 判断此时的状态是否是之前的episode遍历过的
            diff=0
            #判断是否出现过同样的timewindow
            diff_TW=0
            for i in range(0,q_table.shape[0]):

                RemainTimeIndex=q_table.loc[i,'RemainTime']
                RemainingTime=RemainingTimeTotal[RemainTimeIndex]
                CurrentStateRemaingingTime=S[1]
                CRT=len(CurrentStateRemaingingTime)
                RT=len(RemainingTime)

                if CRT!=RT:


                    diff_TW+=1
                    diff+=1

                else:
                    #由于窗口时间是被分成了几段interval存储，所以也要遍历
                    for i in range(0,CRT):

                        CurrentWindow=CurrentStateRemaingingTime[i]
                        ExisintWindow=RemainingTime[i]

                        if CurrentWindow.lower_bound!=ExisintWindow.lower_bound:
                            diff_TW += 1
                            diff+=1
                            break

                        elif CurrentStateRemaingingTime.upper_bound !=ExisintWindow.upper_bound:

                            diff_TW += 1
                            diff+=1
                            break

                        else:

                            pass
                    #判断若窗口全都一样，看看其它状态量是否相同
                    if diff_TW==0:

                        if S[0] != q_table.loc[i, 'Storage']:
                            diff += 1



                        else:

                            if S[2] != q_table.loc[i, 'IncomingTask']:
                                diff+=1

            if diff_TW== q_table.shape[0]:

                RemainingTimeTotal.append(S[1])
                RemainingTimeIndex=len(RemainingTimeTotal)
                new = pd.DataFrame({'Accept': 0,
                                    'Reject': 0,
                                    'Storage': S[0],
                                    'RemainTime': RemainingTimeIndex,
                                    'IncomingTask': S[2]},
                                   index=[0])

            else:

                if diff== q_table.shape[0]:

                    new = pd.DataFrame({'Accept': 0,
                                        'Reject': 0,
                                        'Storage': S[0],
                                        'RemainTime':RemainTimeIndex,
                                        'IncomingTask': S[2]},
                                       index=[0])

                    q_table = q_table.append(new, ignore_index=True)


        else:

            R=0

            S[2] = taskList[0]

    return S, R,q_table

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S[2] == 3:
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0


        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    # 环境更新
        while not is_terminated:

            A = choose_action(S, q_table)   # 选行为
            S_, R,q_table = get_env_feedback(S, A,q_table)  # 实施行为并得到环境的反馈
            q_predict = q_table.loc[S[3], A]    # 估算的(状态-行为)值
            if S_[2] != 3:
                q_target = R + GAMMA * q_table.iloc[S_[3], :].max()   #  实际的(状态-行为)值 (回合没结束)
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.loc[S[3], A] += ALPHA * (q_target - q_predict)  #  q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
