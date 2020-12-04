

# Hyperparameters 
total_steps = 8e6 
num_envs = 32
start_level = 100
num_levels = 200




feature_dim_ = int(256*6) 


checkpoint_file_input = r'.\Week12_Results\checkpoint_impala_stack_background6.pt'
mp4_file_output = r'.\Week12_Results\checkpoint_impala_stack_background6_vid_generalization.mp4'
evaluation_file = r'.\Week12_Results\checkpoint_impala_stack_background6_evaluation_generalization.txt'


#%%

import copy
import torch
import torch.nn as nn
import numpy as np
from utils import make_env, orthogonal_init
import imageio


#%%


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#%%

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

#%% Impala encoder 
# https://github.com/joonleesky/train-procgen-pytorch/blob/master/common/model.py

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 feature_dim
                 ):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=feature_dim)
        # 32*8*8 is 2048

        #self.output_dim = feature_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

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
vec_env = make_env(num_envs, 
				   start_level=start_level, 
				   num_levels=num_levels,
				   use_backgrounds = True,
				   vector_stack = True)

print('Observation space:', vec_env.observation_space)
print('Action space:', vec_env.action_space.n)


#%%

encoder = ImpalaModel(in_channels = 9, feature_dim=feature_dim_)


#%%

trained_policy = Policy(encoder, 
                feature_dim = feature_dim_, 
                num_actions = vec_env.action_space.n)


#%%
trained_policy.load_state_dict(torch.load(checkpoint_file_input))

trained_policy.cuda()
#%%

#%% Running for fixed duration

# Make evaluation environment
eval_env = make_env(num_envs, 
					start_level=start_level, 
					num_levels=num_levels,
					use_backgrounds = True,
					vector_stack = True) 

obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
trained_policy.eval()
for _ in range(512): #Default 512

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
string_eval_1 = 'Average return (fixed simulation lenght): {}'.format(total_reward)
print(string_eval_1)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave(mp4_file_output, frames, fps=25)


#%% Running until all done

# Make evaluation environment
eval_env = make_env(num_envs, 
					start_level=start_level, 
					num_levels=num_levels,
					use_backgrounds = True,
					vector_stack = True) 
obs = eval_env.reset()


reward_matrix = []
still_running = np.ones(num_envs, dtype=bool)
still_running_matrix = []

# Evaluate policy
count = 0
loop_running = True
trained_policy.eval()
while loop_running: 

  # Use policy
  action, log_prob, value = trained_policy.act(obs)

  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  reward_matrix.append(torch.Tensor(reward))
  
  still_running[done] = False #When a simultion is done its index will be False
  still_running_matrix.append(torch.from_numpy(copy.deepcopy(still_running)))

  count += 1
  if still_running.sum() == 0:
      loop_running = False


still_running_matrix = torch.stack(still_running_matrix)
reward_matrix = torch.stack(reward_matrix)


total_reward = 0
for i in range(num_envs):
    
    reward = reward_matrix[:,i]

    boolean = still_running_matrix[:,i]

    total_reward += reward[boolean].sum()

average_reward = total_reward/num_envs

# Calculate average return
string_eval_2 = 'Average return (running until death): {}'.format(average_reward)
print(string_eval_2)
string_eval_3 = "Longest run in number of steps {}".format(count)
print(string_eval_3)


#%%

with open(evaluation_file, 'w') as outfile:
    outfile.write(string_eval_1 + "\n\n\n")
    outfile.write(string_eval_2 + "\n")
    outfile.write(string_eval_3 + "\n")
    