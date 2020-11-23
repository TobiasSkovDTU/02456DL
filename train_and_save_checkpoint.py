

# Hyperparameters 
total_steps = 8e6 
num_envs = 32
start_level = 0
num_levels = 100
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

feature_dim_ = int(256*2) # TA: Experiment with higher


checkpoint_output = r'checkpoint.pt'

#%%


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init


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
env = make_env(num_envs, start_level=start_level, num_levels=num_levels)
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)


#%%


# Define network
encoder = Encoder(in_channels = 3, feature_dim = feature_dim_)
policy = Policy(encoder, 
                feature_dim = feature_dim_, 
                num_actions = env.action_space.n)
policy.cuda()

#%%

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)


#%%
# Run training
obs = env.reset()
step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  # (data collectionfor num_envs*num_steps)
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  #Taking "num_epochs" SGD steps based on same data each iteration (step)
  for epoch in range(num_epochs): 

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)

    
    #num_batches = (num_envs*num_steps)/batch_size 
    # (32*256)/512= 16 default
    for batch in generator: 
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      
      
      #######################################
      ratio = torch.exp(new_log_prob - b_log_prob)
      
      clipped_ratio = ratio.clamp(min=1.0 - eps,
                                  max=1.0 + eps)
      
      policy_reward = torch.min(ratio * b_advantage,
                                  clipped_ratio * b_advantage)

      # Clipped policy objective  = L^CLIP
      pi_loss = -policy_reward.mean()

      #######################################
      
      # Clipped value function objective L^VF
      clipped_value = b_value + \
          (new_value - b_value).clamp(min=-eps, max=eps)
          
      value_loss = 0.5 * torch.max((new_value - b_returns) ** 2, 
                             (clipped_value - b_returns) ** 2).mean()
      
      ########################################

      # Entropy loss          S[Pi_theta](s_t)  (making enough exploration)
      entropy_loss = new_dist.entropy().mean()

      ########################################
       
      
      # Backpropagate losses     L^PPO
      loss = pi_loss + value_coef*value_loss - entropy_coef*entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

  # Update stats
  step += num_envs * num_steps
  print(f'Step: {step}\tMean reward: {storage.get_reward()}')

print('Completed training!')
torch.save(policy.state_dict(), checkpoint_output)

