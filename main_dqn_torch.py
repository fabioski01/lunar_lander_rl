import gymnasium as gym
import os
import matplotlib.pyplot as plt
import json # for dumping debug data
import time # for benchmarking
import numpy as np
from train_torch import Agent # This imports the Agent class from train_torch.py file, which contains the core logic for the agent.

FOLDER_NAME = 'torch_dqn_wind' # change depending on wind or not simulations
# LOAD_MODEL_FILENAME = f'{FOLDER_NAME}/dqn_model' # Change this to change model to load

MODEL_FILENAME_TEMPLATE = f'{FOLDER_NAME}/dqn_model_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.h5'
SCORES_FILENAME_TEMPLATE = f'{FOLDER_NAME}/dqn_scores_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.json'
EPSILON_HISTORY_FILENAME_TEMPLATE = f'{FOLDER_NAME}/dqn_epsilon_history_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.json'
PLOT_FILENAME_TEMPLATE = f'plots/dqn_reward_per_episode_{{episodes}}_eps_{{eps}}_eps_d_{{eps_d}}_bs_{{bs}}_lr_{{lr}}.png'

# Ensure the directory exists
import os
os.makedirs(FOLDER_NAME, exist_ok=True)
os.makedirs('plots', exist_ok=True)

LEARN_EVERY = 4
def train_agent(n_episodes=1500, epsilon=1.0, epsilon_dec=0.995, batch_size=128, lr=0.001, load_latest_model=False):
    print(f"Training a DQN agent on {n_episodes} episodes. Pretrained model = {load_latest_model}")
    
    # Creates an environment
    # WATCH OUT FOR WIND
    env = gym.make("LunarLander-v2", enable_wind = True, wind_power = 15.0, turbulence_power = 1.0)
    # Initializes the DoubleQAgent with specific hyperparameters
    agent = Agent(gamma=0.99,epsilon=epsilon,epsilon_dec=epsilon_dec,lr=lr,mem_size=200000,batch_size=batch_size,epsilon_end=0.01)
    
    # If load_latest_model is set to True, the agent loads a pre-trained model from ddqn_torch_model.h5.
    if load_latest_model:
        agent.load_saved_model(LOAD_MODEL_FILENAME)
        print('Loaded most recent model: {}'.format(LOAD_MODEL_FILENAME))
        
    scores = []
    eps_history = []
    start = time.time()
    for i in range(n_episodes):
        terminated = False
        truncated = False
        score = 0
        state = env.reset()[0]
        steps = 0
        while not (terminated or truncated):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.save(state, action, reward, new_state, terminated)
            state = new_state
            if steps > 0 and steps % LEARN_EVERY == 0:
                agent.learn()
            steps += 1
            score += reward
            
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])

        if (i+1) % 10 == 0 and i > 0:
            # Report expected time to finish the training
            print('Episode {} in {:.2f} min. Expected total time for {} episodes: {:.0f} min. [{:.2f}/{:.2f}]'.format((i+1), 
                                                                                                                      (time.time() - start)/60, 
                                                                                                                      n_episodes, 
                                                                                                                      (((time.time() - start)/i)*n_episodes)/60, 
                                                                                                                      score, 
                                                                                                                      avg_score))
            
        if (i+1) % 100 == 0 and i > 0:
            model_filename = MODEL_FILENAME_TEMPLATE.format(
                episodes=n_episodes,
                eps=epsilon,
                eps_d=epsilon_dec,
                bs=batch_size,
                lr=lr
            )
            scores_filename = SCORES_FILENAME_TEMPLATE.format(
                episodes=n_episodes,
                eps=epsilon,
                eps_d=epsilon_dec,
                bs=batch_size,
                lr=lr
            )
            eps_history_filename = EPSILON_HISTORY_FILENAME_TEMPLATE.format(
                episodes=n_episodes,
                eps=epsilon,
                eps_d=epsilon_dec,
                bs=batch_size,
                lr=lr
            )
            
            print(f"Saving model to: {model_filename}")
            print(f"Saving scores to: {scores_filename}")
            print(f"Saving epsilon history to: {eps_history_filename}")
            
            try:
                agent.save_model(model_filename)
            except Exception as e:
                print(f"Error saving model: {e}")
            
            try:
                with open(scores_filename, "w") as fp:
                    json.dump(scores, fp)
                print(f"Scores saved successfully to: {scores_filename}")
            except Exception as e:
                print(f"Error saving scores JSON: {e}")

            try:
                with open(eps_history_filename, "w") as fp:
                    json.dump(eps_history, fp)
                print(f"Epsilon history saved successfully to: {eps_history_filename}")
            except Exception as e:
                print(f"Error saving epsilon history JSON: {e}")

            # # Save the model every N-th step just in case
            # agent.save_model('ddqn_torch_model.h5')
            # with open("ddqn_torch_dqn_scores_{}.json".format(int(time.time())), "w") as fp:
            #     json.dump(scores, fp)
            # with open("ddqn_torch_eps_history_{}.json".format(int(time.time())), "w") as fp:
            #     json.dump(eps_history, fp)
                
    # Plotting rewards per episode
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_episodes + 1), scores, color='blue', marker='o', linestyle='-', markersize=4)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Reward per Episode (Eps={epsilon}, Eps_D={epsilon_dec}, LR={lr}, BS={batch_size})')
    plt.grid(True)
    
    # Save the plot with detailed filename
    plot_filename = PLOT_FILENAME_TEMPLATE.format(
        episodes=n_episodes,
        eps=epsilon,
        eps_d=epsilon_dec,
        bs=batch_size,
        lr=lr
    )
    plt.savefig(plot_filename)
    plt.show()

    return agent

###############################
# Uncomment to train ##########
###############################
agent = train_agent(load_latest_model=False) # missing lr tweaking


################################
# Visualize the model ##########
################################
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# from IPython.display import clear_output

# # Set path to the model to visualize
# model_to_animate = 'ddqn_torch_model.h5'

# def animate_model(name, atype='single'):
#     env = gym.make("LunarLander-v2", render_mode="rgb_array")
#     agent = DoubleQAgent(gamma=0.99, epsilon=0.0, lr=0.0005, mem_size=200000, batch_size=64, epsilon_end=0.01)
#     agent.load_saved_model(name)
#     state, info = env.reset(seed=12)
#     frame_number = 0
#     for _ in range(5):
#         terminated = False
#         truncated = False
#         while not (terminated or truncated):
#             action = agent.choose_action(state)
#             new_state, reward, terminated, truncated, info = env.step(action)
#             state = new_state
#             clear_output(wait=True)
#             plt.imshow( env.render() )
#             # plt.savefig(f'figures_frames/lunar_lander_frame_{frame_number}.png')
#             # plt.savefig(f'lunar_lander_frame_{state}.png')  # Save each frame as an image
#             frame_number += 1
#             # plt.show()
#         state = env.reset()[0]
#     env.close()

# animate_model(model_to_animate, atype='double')