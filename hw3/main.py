import gym
import torch
from a3_gym_env.envs.pendulum import CustomPendulumEnv

from torch import optim

from Modules import PolicyNetwork, ExperienceReplayBuffer

# [DONE] Task 1: Start by implementing an environment interaction loop. You may refer to homework 1 for inspiration.
# [ ] Task 2: Create and test an experience replay buffer with a random policy, which is the Gaussian distribution with arbitrary (randomly initialized) weights of the policy feed-forward network,receiving state, s, and returning the mean, mu(s) and the log_std, log_stg(s) (natural logarithm of the standard deviation) of actions.  As mentioned above, you can use a state-independent standard variance.
# [ ] Task 3: Make an episode reward processing function to turn one-step rewards into discounted rewards-to-go: R(s_1) = sum_{t=1} gamma^{t-1} r_t, which is the discounted reward, starting from the state, s_1.
# [ ] Task 4: Start the model by implementing a vanilla policy gradient agent, where the gradient ascent stepsare done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go   from each state. Try different step sizes in the gradient ascent.
# [ ] Task 5: Pendulum is a continuous action space environment. Check out the example in `Modules.py` for torch implementation of the Gaussian module.  (if you work in Julia, speak with me regarding the pendulum dynamics in Julia, and Flux for DNNs.)
# [ ] Task 6: Add a feed-forward network for the critic, accepting the state, s=[sin(angle), cos(angle), angular velocity], and returning a scalar for the value of the state, s.
# [ ] Task 7: Implement the generalized advantage, see Eq11-12 in the PPO paper, to be used instead of rewards-to-go.
# [ ] Task 8: Implement the surrogate objective for the policy gradient, see Eq7, without and without clipping.
# [ ] Task 9: Implement the total loss, see Eq9 in the PPO.
# [ ] Task 10: Combine all together to Algorithm 1 in the PPO paper. (In your basic implementation, you can collect data with a single actor, N=1)
# [ ] Task 11: You should see progress with default hyperparameters, but you can try tuning those to see how it will improve your results.

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

    max_step = 1000
    obs = env.reset()
    for _ in range(max_step):
        # get a random action in this environment
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # render already plots the graph for you, no need to use plt
        img = env.render()
        if done:
            obs = env.reset()
    env.close()

def test_experience_relay_buffer():
    env = gym.make("Pendulum-v1-custom")
    # sample hyperparameters
    batch_size = 10000
    epochs = 30
    learning_rate = 1e-2
    hidden_size = 8
    n_layers = 2

    # optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    max_step = 1000
    policy = PolicyNetwork(3, 2).to(device)
    memory = ExperienceReplayBuffer(batch_size)

    for _ in range(max_step):
        # get a random action in this environment
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        memory.push(state, action, next_state, reward)

        gaus_param = policy(state)
        
        print(gaus_param)
        state = next_state
        # render already plots the graph for you, no need to use plt
        img = env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    # interaction_loop()
    test_experience_relay_buffer()
