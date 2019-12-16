from Tutorial.env import ArmENv

from Tutorial.rl import DDPG
MAX_EPISODE=500
MAX_EP_STEPS=200

env=ArmENv()
s_dim=env.state_dim
a_dim=env.aciton_dim
a_bound=env.action_bound

rl=DDPG(a_dim,s_dim,a_bound)

for i in range(MAX_EPISODE):
    s=env.reset()

    for j in range(MAX_EP_STEPS):
        env.render()

        a=rl.choose_action(s)

        s_,r,done=env.step(a)

        rl.store_transtion(s,a,r,s_)

        if rl.memory_full:

            rl.learn()

        s=s_




