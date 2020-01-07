"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import myEnvLocal as myEnv
import globalVariableLocal as globalVariable
import RemainingTimeTotalModule
import time
# env = gym.make('Pendulum-v0').unwrapped
env = myEnv.MyEnv()
EP_MAX = 400 #The maximum nmuber of training episodes
MAX_EP_LEN = 100 #The maximum lenth of each episode
GAMMA = 0.9
A_LR = 0.0001 #learning rate of actor
C_LR = 0.0002 #learning rate of critic
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
N_S = env.observation_space.n
N_A =env.action_space.n
    # env.action_space.n
S_DIM, A_DIM = N_S, N_A

#选择优化方法
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
         ][1]        # choose the method for optimization


class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        # saver = tf.train.Saver(max_to_keep=1)
        # critic
        with tf.variable_scope('critic'):
            #variable_scope下声明共享后，tf.Variable()同名变量指向两个不同变量实体，而tf.get_variable ()同名变量则指向同一个变量实体
            # W1 = np.random.randn(S_DIM,100) * np.sqrt(2 /S_DIM)
            # init1 = tf.constant_initializer(W1)

            # l1 = tf.layers.dense(self.tfs, 100,activation=tf.nn.relu,kernel_initializer=init1,
            # bias_initializer=tf.zeros_initializer)

            l1 = tf.layers.dense(self.tfs, 100, activation=tf.nn.relu)
            l2 = tf.layers.dense(l1, 100, activation=tf.nn.relu)
            # l3 = tf.layers.dense(l2, 300, activation=tf.nn.relu)
            # W2 = np.random.randn(100, 1) * np.sqrt(2 / 100)
            # init2 = tf.constant_initializer(W2)
            self.v = tf.layers.dense(l2, 1)
            # self.v = tf.layers.dense(l1, 1,kernel_initializer=init2,
            # bias_initializer=tf.zeros_initializer)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        # pi, pi_params = self._build_anet('pi', trainable=True)
        # oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.pi, pi_params = self._build_anetDiscrete('pi', trainable=True)
        oldpi, oldpi_params = self._build_anetDiscrete('oldpi', trainable=False)
        # with tf.variable_scope('sample_action'):
        #     # tf.variable_scope()的作用是为了实现变量共享，
        #     # 它和tf.get_variable()来完成变量共享的功能。
        #     self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        #     # choosing action sample_op是一个张量
        #     # 这边pi是一个正态分布，sample(1)
        #     # 就是采一个点（就是选一个动作）
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
            #更新oldp参数
            # zip函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        #[None,]表示行不定，无列
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        #张量拼接函数tf.stack(),range()函数用于创建数字序列变量
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)  # shape=(None, )
        #nd的意思是可以收集n dimension的tensor,
        # 按照indices的格式从params中抽取切片（合并为一个Tensor）
        # indices是一个K维整数Tensor
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob / oldpi_prob

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                # ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, self.pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        a=a.ravel()
        # a= a[:, S_DIM: S_DIM + 1].ravel()
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]
            # [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anetDiscrete(self,name, trainable):

        with tf.variable_scope(name):
            # W3 = np.random.randn(S_DIM, 200) * np.sqrt(2 / S_DIM)
            # init3 = tf.constant_initializer(W3)
            # l_a = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable,
            # kernel_initializer=init3,bias_initializer=tf.zeros_initializer)
            l_a = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            l_a2 = tf.layers.dense(l_a, 200, tf.nn.relu, trainable=trainable)
            # l_a3 = tf.layers.dense(l_a2, 300, tf.nn.relu, trainable=trainable)
            # , kernel_initializer = tf.constant_initializer(np.zeros((4, 200)))
            # W4 = np.random.randn(200, A_DIM) * np.sqrt(2 / 200)
            # init4 = tf.constant_initializer(W4)
            # a_prob = tf.layers.dense(l_a, A_DIM, tf.nn.softmax, trainable=trainable,
            # kernel_initializer=init4,bias_initializer=tf.zeros_initializer)
            a_prob = tf.layers.dense(l_a2, A_DIM, tf.nn.softmax, trainable=trainable)#kernel_initializer=tf.zeros_initializer
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        '''
        用来获取一个名称是‘key’的集合中的所有元素，返回的是一个列表，
        列表的顺序是按照变量放入集合中的先后;   
        scope参数可选，表示的是名称空间（名称域），
        如果指定，就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。
        '''
        return a_prob, params





    # def _build_anet(self, name, trainable):
    #     with tf.variable_scope(name):
    #         l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
    #         # dense ：全连接层
    #         # 相当于添加一个层
    #         mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
    #         #乘二代表建立两次，mu返回的是张量，是该层网络的参数
    #         sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
    #         norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
    #     params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    #     # 从一个集合中取出变量，比如隐藏层的权重
    #     return norm_dist, params

    # def choose_action(self, s):
    #     s = s[np.newaxis, :]
    #     a = self.sess.run(self.sample_op, {self.tfs: s})[0]
    #     return np.clip(a, -2, 2)

    def choose_action_discrete(self, s):  # run by a local
        prob_weights = self.sess.run(self.pi, feed_dict={self.tfs: s[None, :]})
        # if prob_weights[0][0] < 0.1:
        #
        #     # prob_weights[0][0]=prob_weights[0][1]/5+prob_weights[0][0]
        #     prob_weights[0][0]=0.5
        #     prob_weights[0][1]=0.5
        #     # prob_weights[0][1]=prob_weights[0][1]-prob_weights[0][1]/5
        # elif prob_weights[0][1]<0.1:
        #     prob_weights[0][0] = 0.5
        #     prob_weights[0][1] = 0.5
        #
        #     # prob_weights[0][1] = prob_weights[0][0] / 5 + prob_weights[0][1]
        #     # prob_weights[0][0] = prob_weights[0][0] - prob_weights[0][0] / 5
        # else:
        #
        #     pass

        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    def get_v(self, s):
        if s.ndim < 2:#ndim返回的是数组的维度
            s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

# env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []
max_ep_r=0
max_r_episode=[]
globalVariable.initTask()
RemainingTimeTotalModule.initRemainingTimeTotal()
MAX_Record=[]
MAX_Reward=-1000
ax = []                    # 定义一个 x 轴的空列表用来接收动态的数据
ay = []                    # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()
# Rewardrecord=[]
for ep in range(EP_MAX):
    globalVariable.initTasklist()
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    episodeRecord=[]
    for t in range(MAX_EP_LEN):    # in one episode
        a = ppo.choose_action_discrete(s)
        s_, r, done, _ = env.step(a)
        episodeRecord.append([s[1],a])
        buffer_s.append(s)
        buffer_a.append(a)
        # buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        buffer_r.append(r)
        s = s_
        ep_r += r
        if done:

            # print('1')
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:#取从后向前（相反）的元素
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            #将buffer中元素一个个拿出来重新垂直排列
            # buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br) #update critic and actor
            #
            RewardTotal=np.sum(buffer_r)
            if RewardTotal >= MAX_Reward:

                MAX_Record=episodeRecord
                MAX_Reward=RewardTotal

            else:

                pass

            buffer_s, buffer_a, buffer_r = [], [], []

            break

    ax.append(ep)  # 添加 i 到 x 轴的数据中
    ay.append(ep_r)  # 添加 i 的平方到 y 轴的数据中
    plt.clf()  # 清除之前画的图
    plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.pause(0.1)  # 暂停一秒
    plt.ioff()


reward_array=np.array(ay)
np.save('reward2.npy',reward_array)
print('MAX_Record',MAX_Record,'MAX_Reward',MAX_Reward)

