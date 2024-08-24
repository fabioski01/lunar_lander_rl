import matplotlib.pyplot as plt
import json

FOLDER_NAME = 'torch_model'
MODEL_FILENAME = f'{FOLDER_NAME}/ddqn_torch_model_250.h5' # Change this to change the name of the save
LOAD_MODEL_FILENAME = f'{FOLDER_NAME}/ddqn_torch_model.h5' # Change this to change model to load
SCORES_FILENAME_TEMPLATE = f'{FOLDER_NAME}/dqn_scores_{{}}.json'
EPSILON_HISTORY_FILENAME_TEMPLATE = f'{FOLDER_NAME}/epsilon_history_{{}}.json'


def plot_comparison(file_ids, n_episodes):
    plt.figure(figsize=(10, 5))
    
    for file_id in file_ids:
        with open(SCORES_FILENAME_TEMPLATE.format(file_id), "r") as fp:
            scores = json.load(fp)
        plt.plot(range(1, n_episodes + 1), scores, marker='o', linestyle='-', label=f'Epsilon: {file_id}')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode for Different Hyperparameters')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{FOLDER_NAME}/reward_comparison.png')
    plt.show()

# Example usage
# Define file ids corresponding to different hyperparameter configurations
file_ids = ['epsilon_1.0', 'epsilon_0.9', 'epsilon_0.01']
plot_comparison(file_ids, n_episodes=2000)
