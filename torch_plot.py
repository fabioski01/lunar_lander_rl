import matplotlib.pyplot as plt
import json
import os

# Define the hyperparameters for each model you want to compare
eps_values = [1.0, 0.999, 0.9]
n_episodes = 200  # Use the same number of episodes for all models

# Set up folder and file naming conventions
FOLDER_NAME = 'torch_model'
SCORES_FILENAME_TEMPLATE = f'{FOLDER_NAME}/dqn_scores_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.json'

# Define the other hyperparameters for the models
epsilon_dec = 0.995
batch_size = 128
lr = 0.001

# Prepare to plot
plt.figure(figsize=(12, 6))
for eps in eps_values:
    # Generate filename
    scores_filename = SCORES_FILENAME_TEMPLATE.format(
        episodes=n_episodes,
        eps=eps,
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
    plt.plot(range(1, n_episodes + 1), scores, label=f'Eps={eps}', marker='o')

# Finalize the plot
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode for Different Models')
plt.legend()
plt.grid(True)

# Save and show the plot
plot_filename = f'plots/reward_per_episode_comparison.png'
plt.savefig(plot_filename)
print(f"Plot saved to: {plot_filename}")
plt.show()
