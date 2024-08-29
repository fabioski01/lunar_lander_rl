import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Define the hyperparameters for each model you want to compare
# eps_values = [1.0, 0.999, 0.9] # nominal is 1.0
# lr_values = [0.01, 0.001, 0.0001] # nominal is 0.0001
# eps_dec_values = [0.90, 0.995, 0.999] # nominal is 0.995
# batch_size_values = [64, 128, 256] # nominal is 128
n_episodes = 1500  # Use the same number of episodes for all models
window_size = 50   # The window size for calculating the moving average
rolling_window = 100  # The window size for calculating the rolling avera

# # Set up folder and file naming conventions
# FOLDER_NAME = 'torch_dqn'
# SCORES_FILENAME_TEMPLATE = f'{FOLDER_NAME}/dqn_scores_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.json'

#########################################################
# Define the other hyperparameters for the models #######
# Comment the one which is being compared ###############
#########################################################
epsilon = 1.0
epsilon_dec = 0.995
batch_size = 128
lr = 0.001

#########################################################
##################### DQN vs DDQN #######################
#########################################################
# File paths for DQN and DDQN
dqn_folder = 'torch_dqn'
ddqn_folder = 'torch_model'

dqn_filename = f'{dqn_folder}/dqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'
ddqn_filename = f'{ddqn_folder}/ddqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'

# Prepare to plot DQN vs. DDQN
plt.figure(figsize=(12, 6))

# Plot DQN results
if os.path.isfile(dqn_filename):
    with open(dqn_filename, 'r') as fp:
        dqn_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), dqn_scores, label='DQN', marker='o')
else:
    print(f"DQN file not found: {dqn_filename}")

# Plot DDQN results
if os.path.isfile(ddqn_filename):
    with open(ddqn_filename, 'r') as fp:
        ddqn_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), ddqn_scores, label='DDQN', marker='o')
else:
    print(f"DDQN file not found: {ddqn_filename}")

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Reward per Episode DQN vs DDQN\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/dqn_vs_ddqn/reward_per_episode_comparison_DQN_vs_DDQN-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

### Average reward plot
# Prepare to plot average rewards
plt.figure(figsize=(12, 6))

# Calculate and plot average rewards for DQN
if os.path.isfile(dqn_filename):
    avg_dqn_scores = []
    x_ticks = []
    for i in range(0, len(dqn_scores), window_size):
        window_avg = np.mean(dqn_scores[i:i + window_size])
        avg_dqn_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_dqn_scores, label='DQN', marker='o')

# Calculate and plot average rewards for DDQN
if os.path.isfile(ddqn_filename):
    avg_ddqn_scores = []
    x_ticks = []
    for i in range(0, len(ddqn_scores), window_size):
        window_avg = np.mean(ddqn_scores[i:i + window_size])
        avg_ddqn_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_ddqn_scores, label='DDQN', marker='o')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel(f'Average Reward (over {window_size} episodes)')
plt.title(f'Average Reward per {window_size} Episodes: DQN vs DDQN with \n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/dqn_vs_ddqn/reward_avg_per_{window_size}_episodes_comparison_DQN_vs_DDQN-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

##############################
### rolling average
plt.figure(figsize=(12, 6))

# Calculate and plot rolling average rewards for DQN
if os.path.isfile(dqn_filename):
    rolling_avg_dqn_scores = np.convolve(dqn_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(dqn_scores) + 1), rolling_avg_dqn_scores, label=f'DQN (Rolling Avg per {rolling_window} Episodes)', linestyle='--')

# Calculate and plot rolling average rewards for DDQN
if os.path.isfile(ddqn_filename):
    rolling_avg_ddqn_scores = np.convolve(ddqn_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(ddqn_scores) + 1), rolling_avg_ddqn_scores, label=f'DDQN (Rolling Avg per {rolling_window} Episodes)', linestyle='--')

# Finalize the plot for rolling average rewards
plt.xlabel('Episode')
plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)')
plt.title(f'Rolling Average Reward per {rolling_window} Episodes: DQN vs DDQN with \n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the rolling average reward plot
rolling_avg_plot_filename = f'plots/dqn_vs_ddqn/reward_rolling_avg_per_{rolling_window}_episodes_comparison_DQN_vs_DDQN-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(rolling_avg_plot_filename)
print(f"Rolling average reward plot saved to: {rolling_avg_plot_filename}")
plt.show()

#########################################################
##################### EPSILON ###########################
#########################################################
# Prepare to plot
# plt.figure(figsize=(12, 6))
# for eps in eps_values:
#     Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=eps,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue

#     Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     Plot
#     plt.plot(range(1, n_episodes + 1), scores, label=f'Eps={eps}', marker='o')

# Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title(f'Reward per Episode for different models with \n(epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# Save and show the plot
# plot_filename = f'plots/reward_per_episode_comparison_eps-{n_episodes}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ## Average reward plot
# Prepare to plot
# plt.figure(figsize=(12, 6))
# for eps in eps_values:
#     Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=eps,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue

#     Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     Calculate the moving average for each window of 50 episodes
#     avg_scores = []
#     x_ticks = []
#     for i in range(0, len(scores), window_size):
#         window_avg = np.mean(scores[i:i + window_size])
#         avg_scores.append(window_avg)
#         x_ticks.append(i + window_size)  # x values should be at the end of each window

#     Plot
#     plt.plot(x_ticks, avg_scores, label=f'Eps={eps}', marker='o')

# Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Average Reward (over {window_size} episodes)')
# plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# Save and show the plot
# plot_filename = f'plots/reward_avg_per_{window_size}_episodes_comparison_eps-{n_episodes}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

#########################################################
#################### LEARNING RATE ######################
#########################################################
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for lr in lr_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=epsilon,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     # Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue
#     # Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     # Plot
#     plt.plot(range(1, n_episodes + 1), scores, label=f'L.R.={lr}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title(f'Reward per Episode for different models with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/reward_per_episode_comparison_lr-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ### Average reward plot
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for lr in lr_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=epsilon,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     # Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue

#     # Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     # Calculate the moving average for each window of 50 episodes
#     avg_scores = []
#     x_ticks = []
#     for i in range(0, len(scores), window_size):
#         window_avg = np.mean(scores[i:i + window_size])
#         avg_scores.append(window_avg)
#         x_ticks.append(i + window_size)  # x values should be at the end of each window

#     # Plot
#     plt.plot(x_ticks, avg_scores, label=f'L.R.={lr}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Average Reward (over {window_size} episodes)')
# plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/reward_avg_per_{window_size}_episodes_comparison_lr-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

#########################################################
###################### BATCH SIZE #######################
#########################################################
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for batch_size in batch_size_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=epsilon,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     # Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue
#     # Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     # Plot
#     plt.plot(range(1, n_episodes + 1), scores, label=f'B.S.={batch_size}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title(f'Reward per Episode for different models with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/reward_per_episode_comparison_bs-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ### Average reward plot
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for batch_size in batch_size_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=epsilon,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     # Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue

#     # Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     # Calculate the moving average for each window of 50 episodes
#     avg_scores = []
#     x_ticks = []
#     for i in range(0, len(scores), window_size):
#         window_avg = np.mean(scores[i:i + window_size])
#         avg_scores.append(window_avg)
#         x_ticks.append(i + window_size)  # x values should be at the end of each window

#     # Plot
#     plt.plot(x_ticks, avg_scores, label=f'B.S.={batch_size}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Average Reward (over {window_size} episodes)')
# plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/reward_avg_per_{window_size}_episodes_comparison_bs-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

#########################################################
################# EPSILON DECREMENT #####################
#########################################################
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for epsilon_dec in eps_dec_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=epsilon,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     # Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue
#     # Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     # Plot
#     plt.plot(range(1, n_episodes + 1), scores, label=f'eps_dec={epsilon_dec}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title(f'Reward per Episode for different models with \n(epsilon ={epsilon}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/reward_per_episode_comparison_eps_dec-{n_episodes}_eps_{epsilon}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ### Average reward plot
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for epsilon_dec in eps_dec_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=epsilon,
#         eps_d=epsilon_dec,
#         bs=batch_size,
#         lr=lr
#     )
    
#     # Check if file exists
#     if not os.path.isfile(scores_filename):
#         print(f"File not found: {scores_filename}")
#         continue

#     # Load scores
#     try:
#         with open(scores_filename, 'r') as fp:
#             scores = json.load(fp)
#     except Exception as e:
#         print(f"Error loading scores from {scores_filename}: {e}")
#         continue

#     # Calculate the moving average for each window of 50 episodes
#     avg_scores = []
#     x_ticks = []
#     for i in range(0, len(scores), window_size):
#         window_avg = np.mean(scores[i:i + window_size])
#         avg_scores.append(window_avg)
#         x_ticks.append(i + window_size)  # x values should be at the end of each window

#     # Plot
#     plt.plot(x_ticks, avg_scores, label=f'eps_dec={epsilon_dec}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Average Reward (over {window_size} episodes)')
# plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon ={epsilon}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/reward_avg_per_{window_size}_episodes_comparison__eps_dec-{n_episodes}_eps_{epsilon}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()