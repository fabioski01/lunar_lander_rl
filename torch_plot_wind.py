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

# File paths for DDQN models
ddqn_wind_folder = 'torch_DDqn_wind'
ddqn_non_wind_folder = 'torch_DDqn'

ddqn_wind_filename = f'{ddqn_wind_folder}/ddqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'
ddqn_non_wind_filename = f'{ddqn_non_wind_folder}/ddqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'

# File paths for DQN models
dqn_wind_folder = 'torch_dqn_wind'
dqn_non_wind_folder = 'torch_dqn'

dqn_wind_filename = f'{dqn_wind_folder}/dqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'
dqn_non_wind_filename = f'{dqn_non_wind_folder}/dqn_scores_{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.json'

# Prepare to plot DDQN and DQN results
plt.figure(figsize=(12, 6))

# Plot DDQN with wind results
if os.path.isfile(ddqn_wind_filename):
    with open(ddqn_wind_filename, 'r') as fp:
        ddqn_wind_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), ddqn_wind_scores, label='DDQN Wind Scenario', marker='o')
else:
    print(f"DDQN Wind file not found: {ddqn_wind_filename}")

# Plot DDQN without wind results
if os.path.isfile(ddqn_non_wind_filename):
    with open(ddqn_non_wind_filename, 'r') as fp:
        ddqn_non_wind_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), ddqn_non_wind_scores, label='DDQN Non-Wind Scenario', marker='o')
else:
    print(f"DDQN Non-Wind file not found: {ddqn_non_wind_filename}")

# Plot DQN with wind results
if os.path.isfile(dqn_wind_filename):
    with open(dqn_wind_filename, 'r') as fp:
        dqn_wind_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), dqn_wind_scores, label='DQN Wind Scenario', marker='x')
else:
    print(f"DQN Wind file not found: {dqn_wind_filename}")

# Plot DQN without wind results
if os.path.isfile(dqn_non_wind_filename):
    with open(dqn_non_wind_filename, 'r') as fp:
        dqn_non_wind_scores = json.load(fp)
    plt.plot(range(1, n_episodes + 1), dqn_non_wind_scores, label='DQN Non-Wind Scenario', marker='x')
else:
    print(f"DQN Non-Wind file not found: {dqn_non_wind_filename}")

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title(f'Reward per Episode: DQN vs DDQN with and without Wind\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn_wind/reward_per_episode_comparison_DQN_vs_DDQN_with_and_without_wind-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

### Average reward plot
# Prepare to plot average rewards
plt.figure(figsize=(12, 6))

# Calculate and plot average rewards for DDQN with wind scenario
if os.path.isfile(ddqn_wind_filename):
    avg_ddqn_wind_scores = []
    x_ticks = []
    for i in range(0, len(ddqn_wind_scores), window_size):
        window_avg = np.mean(ddqn_wind_scores[i:i + window_size])
        avg_ddqn_wind_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_ddqn_wind_scores, label='DDQN Wind Scenario', marker='o')

# Calculate and plot average rewards for DDQN without wind scenario
if os.path.isfile(ddqn_non_wind_filename):
    avg_ddqn_non_wind_scores = []
    x_ticks = []
    for i in range(0, len(ddqn_non_wind_scores), window_size):
        window_avg = np.mean(ddqn_non_wind_scores[i:i + window_size])
        avg_ddqn_non_wind_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_ddqn_non_wind_scores, label='DDQN Non-Wind Scenario', marker='o')

# Calculate and plot average rewards for DQN with wind scenario
if os.path.isfile(dqn_wind_filename):
    avg_dqn_wind_scores = []
    x_ticks = []
    for i in range(0, len(dqn_wind_scores), window_size):
        window_avg = np.mean(dqn_wind_scores[i:i + window_size])
        avg_dqn_wind_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_dqn_wind_scores, label='DQN Wind Scenario', marker='x')

# Calculate and plot average rewards for DQN without wind scenario
if os.path.isfile(dqn_non_wind_filename):
    avg_dqn_non_wind_scores = []
    x_ticks = []
    for i in range(0, len(dqn_non_wind_scores), window_size):
        window_avg = np.mean(dqn_non_wind_scores[i:i + window_size])
        avg_dqn_non_wind_scores.append(window_avg)
        x_ticks.append(i + window_size)
    plt.plot(x_ticks, avg_dqn_non_wind_scores, label='DQN Non-Wind Scenario', marker='x')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel(f'Average Reward (over {window_size} episodes)')
plt.title(f'Average Reward per {window_size} Episodes: DQN vs DDQN with and without Wind\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/ddqn_wind/reward_avg_per_{window_size}_episodes_comparison_DQN_vs_DDQN_with_and_without_wind-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()

### Rolling average plot
plt.figure(figsize=(12, 6))

# Calculate and plot rolling average rewards for DDQN with wind scenario
if os.path.isfile(ddqn_wind_filename):
    rolling_avg_ddqn_wind_scores = np.convolve(ddqn_wind_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(ddqn_wind_scores) + 1), rolling_avg_ddqn_wind_scores, label=f'DDQN Wind Scenario', linestyle='--')

# Calculate and plot rolling average rewards for DDQN without wind scenario
if os.path.isfile(ddqn_non_wind_filename):
    rolling_avg_ddqn_non_wind_scores = np.convolve(ddqn_non_wind_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(ddqn_non_wind_scores) + 1), rolling_avg_ddqn_non_wind_scores, label=f'DDQN Non-Wind Scenario', linestyle='--')

# Calculate and plot rolling average rewards for DQN with wind scenario
if os.path.isfile(dqn_wind_filename):
    rolling_avg_dqn_wind_scores = np.convolve(dqn_wind_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(dqn_wind_scores) + 1), rolling_avg_dqn_wind_scores, label=f'DQN Wind Scenario', linestyle='--')

# Calculate and plot rolling average rewards for DQN without wind scenario
if os.path.isfile(dqn_non_wind_filename):
    rolling_avg_dqn_non_wind_scores = np.convolve(dqn_non_wind_scores, np.ones(rolling_window)/rolling_window, mode='valid')
    plt.plot(range(rolling_window, len(dqn_non_wind_scores) + 1), rolling_avg_dqn_non_wind_scores, label=f'DQN Non-Wind Scenario', linestyle='--')

# Finalize the plot for rolling average rewards
plt.xlabel('Episode', fontsize=17)
plt.ylabel(f'Rolling Average Reward (over {rolling_window} episodes)', fontsize=17)
plt.title(f'Rolling Average Reward per {rolling_window} Episodes: DQN vs DDQN with and without Wind\n(epsilon={epsilon}, epsilon_decrement={epsilon_dec}, Batch Size={batch_size}, Learning Rate={lr})', fontsize=17)
plt.legend(fontsize=17)
plt.grid(True)

# Save and show the rolling average reward plot
rolling_avg_plot_filename = f'plots/ddqn_wind/reward_rolling_avg_per_{rolling_window}_episodes_comparison_DQN_vs_DDQN_with_and_without_wind-{n_episodes}_eps_{epsilon}_eps_d_{epsilon_dec}_bs_{batch_size}_lr_{lr}.png'
plt.savefig(rolling_avg_plot_filename, bbox_inches='tight')
print(f"Rolling average reward plot saved to: {rolling_avg_plot_filename}")
plt.show()
