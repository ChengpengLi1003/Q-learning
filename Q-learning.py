import gym
import random 
import numpy as np 
class base_q_learning():#定义Q learning类
    def __init__(
        self,
        env,#环境
        #num_steps_per_episode=200,#每一轮的训练步数？
        num_steps_per_episode=200,
        gamma=0.5,#折扣因子
        alpha=0.1,#更新步长
        epsilon=0.1#贪婪算法的贪婪系数
    ):
        #初始化参数
        self.env = env
        self.num_steps_per_episode = num_steps_per_episode
        self.state_num = self.env.observation_space.n
        #print(f"action_space: {self.env.action_space.n}")
        self.action_num = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        # 初始化Q表
        self.q = np.zeros((self.state_num, self.action_num))
        self.cur_state = 0
        self.avg_returns = []
    #根据贪婪算法采取动作
    def select_action(self):
        ran_num = random.random()
        if ran_num <= self.epsilon:
            # epsilon randomly choose action
            action = random.randrange(self.action_num)
            return int(action)
        
        else:
            # greedily choose action
            # print(f"type_state: {type(self.cur_state)}")
            action = self.argmax(self.cur_state)
            return int(action)
    #与环境进行交互，更新Q表，计算收益
    def train(self, epochs):
        for epoch in range(epochs):
            # reset env
            env.render()  # 显示图形界面
            self.cur_state = self.env.reset()
            for i in range(self.num_steps_per_episode):
                a = self.select_action()
                next_s, reward, done, info = self.env.step(a)
                if done:
                    break
                self.update_q_table(self.cur_state, a, next_s, reward)
                self.cur_state = next_s
            avg_return = self.evaluate()
            self.avg_returns.append(avg_return)
            print(f"epoch: {epoch}, avg_return: {avg_return}")
    #计算当前Q表对应贪心策略的非折扣累计回报和
    def evaluate(self):
        q = self.q
        env = self.env
        s = env.reset()
        avg_return = 0.0 
        for i in range(self.num_steps_per_episode):
            a = np.argmax(q[s]) 
            next_s, reward, done, info = env.step(a)
            avg_return += reward
            if done: 
                break
            s = int(next_s)
        return avg_return
    #根据Qlearning算法更新Q表
    def update_q_table(self, s, a, next_s, r):        # update q value
        s = int(s)
        a = int(a)
        next_s = int(next_s)
        q_target = r + self.gamma * np.max(self.q[int(next_s)])
        #print("discount_return",q_target)
        self.q[s][a] = self.q[s][a] + self.alpha * (q_target - self.q[s][a])
    #选取能够使得当前状态s下Q值最大的动作号
    def argmax(self, s):
        s = int(s)
        if np.count_nonzero(self.q[s]) == 0:
            action = random.randrange(self.action_num)
        else:
            action = np.argmax(self.q[s])
        return action
#用于测试的主函数，需要输入环境，训练轮数；
if __name__ == '__main__':
    epochs = 1000
    env = gym.make('CliffWalking-v0')  # 生成实验环境，这里选取悬崖寻路环境CliffWalking-v0，可供选取的环境还包括FrozenLake-v0，Roulette-v0，FrozenLake8x8-v0，Taxi-v3
    env.reset()  # 重置环境
    agent = base_q_learning(env)#输入环境
    agent.train(epochs)#与环境交互，训练Q表
    env.close()  # 关掉环境