import torch
import torchvision
import matplotlib.pyplot as plt
from procgen import ProcgenEnv
import gym

#%% Procgen enviroment


env = ProcgenEnv(
		num_envs=32,
		env_name='starpilot',
		start_level=0,
		num_levels=100,
		distribution_mode='easy',
		use_backgrounds=False,
		restrict_themes=True,
		render_mode='rgb_array',
		rand_seed=0
	)



#%%

obs = env.reset()['rgb']

print(obs.shape)

#%%

grid = torchvision.utils.make_grid(
    torch.Tensor(obs).permute(0, 3, 1, 2)).permute(1, 2, 0)/255.

plt.imshow(grid)

#%%

#%%

env = gym.make('procgen:procgen-coinrun-v0')
obs = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        break

