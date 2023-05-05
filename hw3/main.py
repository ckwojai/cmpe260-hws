import gym
import torch
from a3_gym_env.envs.pendulum import CustomPendulumEnv
import numpy as np

from torch import optim

from Modules import PolicyNetwork, ExperienceReplayBuffer, ValueNetwork
from torch.distributions import MultivariateNormal
from torch.optim import Adam

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
    """"
    * Start by implementing an environment interaction loop. You may refer to homework 1 for inspiration. 
    """
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

def get_action(mean, std=0.5):
    # Create the covariance matrix
    action_dim = 1
    cov_var = torch.full(size=(action_dim,), fill_value=0.5)
    cov_mat = torch.diag(cov_var)
    dist = MultivariateNormal(mean.cpu(), cov_mat.cpu())

    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.detach().numpy(), log_prob.detach()

def test_experience_relay_buffer():
    """
    * Create and test an experience replay buffer with a random policy, which is the 
    Gaussian distribution with arbitrary (randomly initialized) weights of the policy feed-forward network,
    receiving state, s, and returning the mean, mu(s) and the log_std, log_stg(s) 
    (natural logarithm of the standard deviation) of actions.  As mentioned above, you can use 
    a state-independent standard variance.
    """
    env = gym.make("Pendulum-v1-custom")
    # Hyperparameters
    max_step = 200
    batch_size = 10
    max_buffer_size = 100
    num_replay = 1

    # Policy Network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy = PolicyNetwork(input_size, output_size).to(device)


    # Replay Buffer
    memory = ExperienceReplayBuffer(max_buffer_size)

    # Env
    state = env.reset()
    state = torch.from_numpy(state).to(device)
    for i in range(max_step):
        # get a random action in this environment
        # action = env.action_space.sample()
        mean = policy(state)
        action, _  = get_action(mean)

        # Take a step
        obs, reward, done, info = env.step(action)
        next_state = torch.from_numpy(obs).to(device)
        reward = torch.tensor([reward], dtype=torch.float32, device=device)

        memory.push(state, action, next_state, reward)

        if len(memory) > batch_size:
            for _ in range(num_replay):
                # sample from replay buffer
                transitions = memory.sample(batch_size)
                # uses the above transitions to optimize the policy
                print(transitions)

        state = next_state
        if done:
            obs = env.reset()
    env.close()


def compute_rtgs(batch_rews, gamma=0.95):
  # The rewards-to-go (rtg) per episode per batch to return.
  # The shape will be (num timesteps per episode)
  batch_rtgs = []
  # Iterate through each episode backwards to maintain same order
  # in batch_rtgs
  for ep_rews in reversed(batch_rews):
    discounted_reward = 0 # The discounted reward so far
    for rew in reversed(ep_rews):
      discounted_reward = rew + discounted_reward * gamma
      batch_rtgs.insert(0, discounted_reward)
  # Convert the rewards-to-go into a tensor
  batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, requires_grad=True)
  return batch_rtgs

def rollout(env, policy, batch_size):
    # Batch data
    batch_obs = []             # batch observations
    batch_acts = []            # batch actions
    batch_log_probs = []       # log probs of each action
    batch_rews = []            # batch rewards
    batch_rtgs = []            # batch rewards-to-go
    batch_lens = []            # episodic lengths in batch

    t = 0
    max_ep_t = batch_size
    while t < batch_size:
        ep_rews = []
        obs = env.reset()
        done = False
        for ep_t in range(max_ep_t):
            t += 1
            batch_obs.append(obs)
            mean = policy(torch.from_numpy(obs).to(device))
            action, log_prob  = get_action(mean)

            obs, reward, done, info = env.step(action)
            ep_rews.append(reward)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            if done:
                break
        batch_lens.append(ep_t+1)
        batch_rews.append(ep_rews)

    # Reshape data as tensors in the shape specified before returning
    batch_obs = torch.tensor(batch_obs, dtype=torch.float, requires_grad=True)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float, requires_grad=True)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, requires_grad=True)
    # ALG STEP #4
    batch_rtgs = compute_rtgs(batch_rews)
    # Return the batch data
    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens 

def test_reward_to_go():
    """
    * Make an episode reward processing function to turn one-step rewards into discounted rewards-to-go:
    R(s_1) = sum_{t=1} gamma^{t-1} r_t, which is the discounted reward, starting from the state, s_1.
    """
    env = gym.make("Pendulum-v1-custom")
    batch_size = 1000
    # Policy Network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy = PolicyNetwork(input_size, output_size).to(device)

    batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = rollout(env, policy, batch_size)
    print(batch_rtgs)


def test_vanilla_gradient_ascent():
    """
    * Start the model by implementing a vanilla policy gradient agent, where the gradient ascent steps
    are done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go   
    from each state. Try different step sizes in the gradient ascent.  
    """
    env = gym.make("Pendulum-v1-custom")

    epoch = 100
    learning_rate = 0.95

    batch_size = 1000
    # Policy Network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy = PolicyNetwork(input_size, output_size).to(device)
    optimizer =  Adam(policy.parameters(), lr=learning_rate)

    for i in range(epoch):
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = rollout(env, policy, batch_size)
        # average of log-likelihood, weighted by rewards-to-go

        loss = -(batch_rtgs * batch_log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
        print(f"{i}: {loss}")

def test_critic():
    """
    * Add a feed-forward network for the critic, accepting the state, 
    s=[sin(angle), cos(angle), angular velocity], and returning a scalar for the value of the state, s.
    """
    env = gym.make("Pendulum-v1-custom")

    epoch = 100
    learning_rate = 0.95

    batch_size = 1000
    # Policy Network
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    policy = PolicyNetwork(input_size, output_size).to(device)
    critic = ValueNetwork(input_size, 1)
    optimizer =  Adam(policy.parameters(), lr=learning_rate)

    for i in range(epoch):
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = rollout(env, policy, batch_size)
        # average of log-likelihood, weighted by rewards-to-go

        loss = -(batch_rtgs * batch_log_probs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step
        print(f"{i}: {loss}")

if __name__ == "__main__":
    # interaction_loop() # Task 1
    # test_experience_relay_buffer() # Task 2
    # test_reward_to_go() # Task 3
    # test_vanilla_gradient_ascent() # Task 4
    # Checked Out Module # Task 5
    test_critic()# Task 6
