

# Hyperparameters 
total_steps = 8e4 #Default 8e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

feature_dim_ = int(256*3) 


checkpoint_file_input = r'.\Week10_Results\checkpoint3.pt'
mp4_file_output = r'.\Week10_Results\vid_checkpoint3.mp4'


#%%


import torch
import torch.nn as nn
from utils import make_env, orthogonal_init


#%%

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#%%

class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)

#%%

class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

#%%

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


#%%

encoder = Encoder(in_channels = 3, feature_dim = feature_dim_)


#%%

trained_policy = Policy(encoder, 
                feature_dim = feature_dim_, 
                num_actions = env.action_space.n)

#%%
trained_policy.load_state_dict(torch.load(checkpoint_file_input))

trained_policy.cuda()

#%%
import imageio

# Make evaluation environment
eval_env = make_env(1, start_level=num_levels, num_levels=num_levels) #Default: num_envs (not 1)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
trained_policy.eval()
for _ in range(512 * 4): #Default 512

  # Use policy
  action, log_prob, value = trained_policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave(mp4_file_output, frames, fps=25)