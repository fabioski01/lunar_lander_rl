import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Define the hyperparameters for each model you want to compare
eps_values = [1.0, 0.999, 0.9] # nominal is 1.0
lr_values = [0.01, 0.001, 0.0001] # nominal is 0.0001
eps_dec_values = [0.90, 0.995, 0.999] # nominal is 0.995
batch_size_values = [64, 128, 256] # nominal is 128
n_episodes = 1500  # Use the same number of episodes for all models
window_size = 50   # The window size for calculating the moving average
rolling_window = 100  # The window size for calculating the rolling avera

# Set up folder and file naming conventions
FOLDER_NAME = 'torch_DDqn'
SCORES_FILENAME_TEMPLATE = f'{FOLDER_NAME}/ddqn_scores_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.json'

#########################################################
# Define the other hyperparameters for the models #######
# Comment the one which is being compared ###############
#########################################################

epsilon = 1.0
lr = 0.001
# batch_size = 128
epsilon_dec = 0.995

#########################################################
##################### EPSILON ###########################
#########################################################
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for eps in eps_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=eps,
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
#     plt.plot(range(1, n_episodes + 1), scores, label=f'Eps={eps}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title(f'Reward per Episode for different models with \n(epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/ddqn/ddqn_epsilon_comparison/reward_per_episode_comparison_eps-{n_episodes}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ### Average reward plot
# # Prepare to plot
# plt.figure(figsize=(12, 6))
# for eps in eps_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=eps,
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
#     plt.plot(x_ticks, avg_scores, label=f'Eps={eps}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Average Reward (over {window_size} episodes)')
# plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/ddqn/ddqn_epsilon_comparison/reward_avg_per_{window_size}_episodes_comparison_eps-{n_episodes}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ##############################
# ### rolling average
# plt.figure(figsize=(12, 6))

# plt.figure(figsize=(12, 6))
# for eps in eps_values:
#     # Generate filename
#     scores_filename = SCORES_FILENAME_TEMPLATE.format(
#         episodes=n_episodes,
#         eps=eps,
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

#     # Calculate the rolling average of rewards
#     rolling_avg_scores = np.convolve(scores, np.ones(rolling_window)/rolling_window, mode='valid')

#     # Plot the rolling average
#     plt.plot(range(rolling_window, len(scores) + 1), rolling_avg_scores, label=f'epsilon={eps}', linestyle='--')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)')
# plt.title(f'Rolling Reward Average per {rolling_window} Episodes: eps comparison with \n(epsilon_dec ={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/ddqn/ddqn_epsilon_comparison/reward_rolling_avg_per_{rolling_window}_episodes_comparison_eps-{n_episodes}_eps_dec_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
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
#     plt.plot(range(1, n_episodes + 1), scores, label=f'Learning Rate={lr}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title(f'Reward per Episode for different models with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/ddqn/ddqn_lr_comparison/reward_per_episode_comparison_lr-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}.png'
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
#     plt.plot(x_ticks, avg_scores, label=f'Learning Rate={lr}', marker='o')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Average Reward (over {window_size} episodes)')
# plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/ddqn/ddqn_lr_comparison/reward_avg_per_{window_size}_episodes_comparison_lr-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ##############################
# ### rolling average
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

#     # Calculate the rolling average of rewards
#     rolling_avg_scores = np.convolve(scores, np.ones(rolling_window)/rolling_window, mode='valid')

#     # Plot the rolling average
#     plt.plot(range(rolling_window, len(scores) + 1), rolling_avg_scores, label=f'Learning Rate={lr}', linestyle='--')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)')
# plt.title(f'Rolling Reward Average per {rolling_window} Episodes: Learning Rate comparison with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'ddqn/ddqn_lr_comparison/reward_rolling_avg_per_{rolling_window}_episodes_comparison_lr-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

#########################################################
###################### BATCH SIZE #######################
#########################################################
# Prepare to plot
plt.figure(figsize=(12, 6))
for batch_size in batch_size_values:
    # Generate filename
    scores_filename = SCORES_FILENAME_TEMPLATE.format(
        episodes=n_episodes,
        eps=epsilon,
        eps_d=epsilon_dec,
        bs=batch_size,
        lr=lr
    )
    
    # Check if file exists
    if not os.path.isfile(scores_filename):
        print(f"File not found: {scores_filename}")
        continue
    # Load scores
    try:
        with open(scores_filename, 'r') as fp:
            scores = json.load(fp)
    except Exception as e:
        print(f"Error loading scores from {scores_filename}: {e}")
        continue

    # Plot
    plt.plot(range(1, n_episodes + 1), scores, label=f'Batch Size={batch_size}', marker='o')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Reward per Episode for different models with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn/ddqn_bs_comparison/reward_per_episode_comparison_bs-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

### Average reward plot
# Prepare to plot
plt.figure(figsize=(12, 6))
for batch_size in batch_size_values:
    # Generate filename
    scores_filename = SCORES_FILENAME_TEMPLATE.format(
        episodes=n_episodes,
        eps=epsilon,
        eps_d=epsilon_dec,
        bs=batch_size,
        lr=lr
    )
    
    # Check if file exists
    if not os.path.isfile(scores_filename):
        print(f"File not found: {scores_filename}")
        continue

    # Load scores
    try:
        with open(scores_filename, 'r') as fp:
            scores = json.load(fp)
    except Exception as e:
        print(f"Error loading scores from {scores_filename}: {e}")
        continue

    # Calculate the moving average for each window of 50 episodes
    avg_scores = []
    x_ticks = []
    for i in range(0, len(scores), window_size):
        window_avg = np.mean(scores[i:i + window_size])
        avg_scores.append(window_avg)
        x_ticks.append(i + window_size)  # x values should be at the end of each window

    # Plot
    plt.plot(x_ticks, avg_scores, label=f'Batch Size={batch_size}', marker='o')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel(f'Average Reward (over {window_size} episodes)')
plt.title(f'Reward Average per {window_size} Episodes with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn/ddqn_bs_comparison/reward_avg_per_{window_size}_episodes_comparison_bs-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

##############################
### rolling average
plt.figure(figsize=(12, 6))

for batch_size in batch_size_values:
    # Generate filename
    scores_filename = SCORES_FILENAME_TEMPLATE.format(
        episodes=n_episodes,
        eps=epsilon,
        eps_d=epsilon_dec,
        bs=batch_size,
        lr=lr
    )
    
    # Check if file exists
    if not os.path.isfile(scores_filename):
        print(f"File not found: {scores_filename}")
        continue
    # Load scores
    try:
        with open(scores_filename, 'r') as fp:
            scores = json.load(fp)
    except Exception as e:
        print(f"Error loading scores from {scores_filename}: {e}")
        continue

    # Calculate the rolling average of rewards
    rolling_avg_scores = np.convolve(scores, np.ones(rolling_window)/rolling_window, mode='valid')

    # Plot the rolling average
    plt.plot(range(rolling_window, len(scores) + 1), rolling_avg_scores, label=f'Batch Size={batch_size}', linestyle='--')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)')
plt.title(f'Rolling Reward Average per {rolling_window} Episodes: Batch Size comparison with \n(epsilon ={epsilon}, epsilon_decrement={epsilon_dec}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn/ddqn_bs_comparison/reward_rolling_avg_per_{rolling_window}_episodes_comparison_bs-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

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
# plot_filename = f'plots/ddqn/ddqn_eps_dec_comparison/reward_per_episode_comparison_eps_dec-{n_episodes}_eps_{epsilon}_bs_{batch_size}_lr_{lr}.png'
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
# plot_filename = f'plots/ddqn/ddqn_eps_dec_comparison/reward_avg_per_{window_size}_episodes_comparison__eps_dec-{n_episodes}_eps_{epsilon}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()

# ##############################
# ### rolling average
# plt.figure(figsize=(12, 6))

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

#     # Calculate the rolling average of rewards
#     rolling_avg_scores = np.convolve(scores, np.ones(rolling_window)/rolling_window, mode='valid')

#     # Plot the rolling average
#     plt.plot(range(rolling_window, len(scores) + 1), rolling_avg_scores, label=f'epsilon_dec={epsilon_dec}', linestyle='--')

# # Finalize the plot
# plt.xlabel('Episode')
# plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)')
# plt.title(f'Rolling Reward Average per {rolling_window} Episodes: eps_dec comparison with \n(epsilon ={epsilon}, Batch Size={batch_size}, Learning Rate={lr})')
# plt.legend()
# plt.grid(True)

# # Save and show the plot
# plot_filename = f'plots/ddqn/ddqn_eps_dec_comparison/reward_rolling_avg_per_{rolling_window}_episodes_comparison_eps_dec-{n_episodes}_eps_{epsilon}_bs_{batch_size}_lr_{lr}.png'
# plt.savefig(plot_filename)
# print(f"Plot saved to: {plot_filename}")
# plt.show()