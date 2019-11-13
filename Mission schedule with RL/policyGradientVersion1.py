#First set up the policy network


import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Constants
GAMMA = 0.9


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions

        '''
        nn.Linear 线性变换，加权重，加偏置.y=Ax+b
        '''

        self.linear1 = nn.Linear(num_inputs, hidden_size)

        self.linear2 = nn.Linear(hidden_size, num_actions)


        '''
        torch.optim是一个实现了多种优化算法的包
        为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态
        ，并能够根据计算得到的梯度来更新参数
        Adam Adam 是一种可以替代传统随机梯度下降（SGD）过程的一阶优化算法，
        它能基于训练数据迭代地更新神经网络权重
        要构建一个优化器optimizer，
        你必须给它一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。
        然后，您可以指定程序优化特定的选项，例如学习速率，权重衰减等
        '''

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):

        x = F.relu(self.linear1(state))#Relu: f(x)=max(0,x)
        x = F.softmax(self.linear2(x), dim=1) #softmax层，将输入转换成概率输出。
        return x

    def get_action(self, state):

        ''''
        torch.autograd.Variable用来包裹张量并记录应用的操作。
        Variable可以看作是对Tensor对象周围的一个薄包装，也包含了和张量相关的梯度，
        以及对创建它的函数的引用。 此引用允许对创建数据的整个操作链进行回溯。
        需要BP的网络都是通过Variable来计算的。如果Variable是由用户创建的，
        则其grad_fn将为None，我们将这些对象称为叶子Variable。
        '''


        state = torch.from_numpy(state).float().unsqueeze(0)#转化为Torch的格式
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
#The update function


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []

    #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
    # 然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。

    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

#The main loop

def main():
    env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)

    max_episode_num = 5000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []

        for steps in range(max_steps):
            env.render()
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,
                                                                                                              np.round(
                                                                                                                  np.sum(
                                                                                                                      rewards),
                                                                                                                  decimals=3),
                                                                                                              np.round(
                                                                                                                  np.mean(
                                                                                                                      all_rewards[
                                                                                                                      -10:]),
                                                                                                                  decimals=3),
                                                                                                              steps))
                break

            state = new_state

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()


if __name__ == "__main__":

    main()