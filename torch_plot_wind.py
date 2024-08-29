import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Define the hyperparameters for each model you want to compare
# eps_values = [1.0, 0.999, 0.9] # nominal is 1.0
# lr_values = [0.01, 0.001, 0.0001] # nominal is 0.0001
# eps_dec_values = [0.90, 0.995, 0.999] # nominal is 0.995
# batch_size_values = [64, 128, 256] # nominal is 128
# Define the hyperparameters for each model
n_episodes = 1500  # Number of episodes
window_size = 50   # Window size for moving average
rolling_window = 100  # Window size for rolling average

# Define other hyperparameters
epsilon = 1.0
epsilon_dec = 0.995
batch_size = 128
lr = 0.001

# File paths for wind and non-wind models
wind_folder = 'torch_DDqn_wind'
non_wind_folder = 'torch_model'

wind_filename = f'{wind_folder}/ddqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'
non_wind_filename = f'{non_wind_folder}/ddqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'

# Prepare to plot wind vs. non-wind results
plt.figure(figsize=(12, 6))

# Plot wind results
if os.path.isfile(wind_filename):
    with open(wind_filename, 'r') as fp:
        wind_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), wind_scores, label='Wind Scenario', marker='o')
else:
    print(f"Wind file not found: {wind_filename}")

# Plot non-wind results
if os.path.isfile(non_wind_filename):
    with open(non_wind_filename, 'r') as fp:
        non_wind_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), non_wind_scores, label='Non-Wind Scenario', marker='o')
else:
    print(f"Non-Wind file not found: {non_wind_filename}")

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Reward per Episode: Wind vs Non-Wind Scenario\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn_wind/reward_per_episode_comparison_Wind_vs_NonWind-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

### Average reward plot
# Prepare to plot average rewards
plt.figure(figsize=(12, 6))

# Calculate and plot average rewards for wind scenario
if os.path.isfile(wind_filename):
    avg_wind_scores = []
    x_ticks = []
    for i in range(0, len(wind_scores), window_size):
        window_avg = np.mean(wind_scores[i:i + window_size])
        avg_wind_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_wind_scores, label='Wind Scenario', marker='o')

# Calculate and plot average rewards for non-wind scenario
if os.path.isfile(non_wind_filename):
    avg_non_wind_scores = []
    x_ticks = []
    for i in range(0, len(non_wind_scores), window_size):
        window_avg = np.mean(non_wind_scores[i:i + window_size])
        avg_non_wind_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_non_wind_scores, label='Non-Wind Scenario', marker='o')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel(f'Average Reward (over {window_size} episodes)')
plt.title(f'Average Reward per {window_size} Episodes: Wind vs Non-Wind Scenario\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn_wind/reward_avg_per_{window_size}_episodes_comparison_Wind_vs_NonWind-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

### Rolling average plot
plt.figure(figsize=(12, 6))

# Calculate and plot rolling average rewards for wind scenario
if os.path.isfile(wind_filename):
    rolling_avg_wind_scores = np.convolve(wind_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(wind_scores) + 1), rolling_avg_wind_scores, label=f'Wind Scenario (Rolling Avg per {rolling_window} Episodes)', linestyle='--')

# Calculate and plot rolling average rewards for non-wind scenario
if os.path.isfile(non_wind_filename):
    rolling_avg_non_wind_scores = np.convolve(non_wind_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(non_wind_scores) + 1), rolling_avg_non_wind_scores, label=f'Non-Wind Scenario (Rolling Avg per {rolling_window} Episodes)', linestyle='--')

# Finalize the plot for rolling average rewards
plt.xlabel('Episode')
plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)')
plt.title(f'Rolling Average Reward per {rolling_window} Episodes: Wind vs Non-Wind Scenario\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the rolling average reward plot
rolling_avg_plot_filename = f'plots/ddqn_wind/reward_rolling_avg_per_{rolling_window}_episodes_comparison_Wind_vs_NonWind-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
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