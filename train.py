import torch
import numpy as np
from game import Game
from matplotlib import pyplot as plt


def main():
    env = Game(dim=4)
    agent = Basic()
    #rewards = []
    losses = []
    for _ in range(10):
        env.reset()
        state = env.state / 17
        i = 0
        done = False
        while not done:
            actions = agent.predict(torch.tensor(state, dtype=torch.float32).reshape(1, -1))
            next_state, reward, done = env.step(torch.argmax(actions).item())
            losses.append(-torch.log(torch.max(actions)) * reward)
            i += 1
            state = next_state
            #rewards.append(reward)
            print(i)
            print(env) 
            #x = input()
            if env.total_reward <= -2:
                break
        loss = sum(losses)/len(losses)
        #print(losses)
        print(loss)
        loss.backward()
        
    #plt.plot(rewards)
    #plt.show()


class Basic:
    def __init__(self, input_size=16):
        g = torch.Generator().manual_seed(67584) # reproducibility
        self.w1 = torch.randn(input_size, 32)
        self.b1 = torch.randn(1, 32)
        
        self.w2 = torch.randn(32, 16)
        self.b2 = torch.randn(1, 16)

        self.w3 = torch.randn(16, 8)
        self.b3 = torch.randn(1, 8)

        self.w4 = torch.randn(8, 4)
        self.b4 = torch.randn(1, 4)

        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]
        self.num_params = sum([p.nelement() for p in self.params])
        for p in self.params:
            p.requires_grad=True
    
    def predict(self, input):
        #print(input.dtype, self.w1.dtype, self.b1.dtype)
        x = input @ self.w1 + self.b1
        x = torch.nn.ReLU()(x)
        x = x @ self.w2 + self.b2
        x = torch.nn.ReLU()(x)
        x = x @ self.w3 + self.b3
        x = torch.nn.ReLU()(x)
        x = x @ self.w4 + self.b4
        x = torch.nn.Softmax()(x)
        return x



if __name__ == '__main__':
    main()