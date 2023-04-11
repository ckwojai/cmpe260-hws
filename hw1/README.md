# CMPE260 Assignmnet 1
The goal of this assignment is for you to become familiar with gym environment, 
apply vae deep learning algorithm implementation,
and see how the training/inference looks like in code.

### Goals 
* explore the gym framework for training rl agents.
* apply your knowledge of VAE to learn image generation.
* train generative models to produce sample pixel observation images from gym environments.

### What to submit
* your `train_vae.py`.
* a doc with generated images and answers to questions in activities.

### Environment
[OpenAI's Gym](https://gym.openai.com/) is a framework for training reinforcement 
learning agents. It provides a set of environments and a
standardized interface for interacting with those.   
In this assignment we will use the [CartPole](https://gym.openai.com/envs/CartPole-v1/) environment from gym.

### Installation

#### Using conda (recommended)    
1. [Install Anaconda](https://www.anaconda.com/products/individual)

2. Create the env    
`conda create a1 python=3.8` 

3. Activate the env     
`conda activate a1`    

4. install torch ([steps from pytorch installation guide](https://pytorch.org/)):    
- if you don't have an nvidia gpu or don't want to bother with cuda installation:    
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`    
  
- if you have an nvidia gpu and want to use it:    
[install cuda](https://docs.nvidia.com/cuda/index.html)   
install torch with cuda:   
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. other dependencies   
`conda install -c conda-forge matplotlib gym opencv pyglet`

#### Using pip
`python3 -m pip install -r requirements.txt`

### Code
`MyVAE.py` - your VAE model   
`train_vae.py` - script to collect pixel observations from gym environments using a random policy and train a vae model     
`sample_vae.py` - samples from a vae trained by train_vae.py    


### Activities

1. Finish the `__init__()` in `MyVAE.py` model.
At this point this is not really a VAE yet, but you should be able 
to train the model. Run `train_vae.py` to train. 
Then, run `sample_vae.py` to generate a few images with your model.   
*Note: you can run `MyVAE.py` to quickly test if your model is working.      
Save two generated images.   
What model components are used in the forward pass and in sampling?    


2. By default, the model behaves as an autoencoder. Upgrade it to 
VAE by modifying `forward()`, `encode()`, and `reparameterize()` 
in `MyVAE.py`.   
Train and save two generated images.      
Describe the difference between the AE and VAE models.   
What is the reparametrization trick?


3. Update the `train_vae.py` to reset 
the environment after the first 20 observations from each episode.    
Train and save two generated images.   
when does the cartpole environment return done=True?

 
4. update the `train_vae.py` train vae on 
observations with a custom angle range. Pick some max and min vales for image observations that
will make generated observations look different from the previous outputs. Don't use states that 
too far from the initialization state, so that the sampling doesn't take too long.    
Train and save two generated images.


5. pick [some other gym environment]((https://gym.openai.com/envs/#classic_control)) 
(environments outside the classical control may require you to install additional libraries) 
and train vae on it.    
Train and save two generated images.


Have fun!