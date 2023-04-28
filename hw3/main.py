import gym
import torch

from torch import optim

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

def interaction_loop():
    env = gym.make("Pendulum-v1-custom")
    # sample hyperparameters
    batch_size = 10000
    epochs = 30
    learning_rate = 1e-2
    hidden_size = 8
    n_layers = 2

    # optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    obs = env.reset()
    max_step = 1000
    for _ in range(max_step):
        # get a random action in this environment
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        # render already plots the graph for you, no need to use plt
        img = env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    interaction_loop()