import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# create environment called LunarLander-v2
env = gym.make("LunarLander-v2", render_mode="human")

# Load the trained model
model_name = "ppo-LunarLander-v2"

try:
    # Try to load the model
    model = PPO.load(model_name)
except FileNotFoundError:
    print(f"Model {model_name} not found. Please ensure the model is saved correctly.")

# Use Monitor for evaluation environment
eval_env = Monitor(gym.make("LunarLander-v2"))

# Evaluate the policy if the model is loaded
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Reset the environment to start a new episode
observation, info = env.reset()

# Use the trained model to make decisions instead of taking random actions. 5000 ri
for _ in range(5000):  # or however many steps you want to run
    # Use the trained model to predict the next action
    action, _states = model.predict(observation, deterministic=True)
    
    # Take the action in the environment
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Print out the details
    # 0=nothing, 1=fire left engine, 2=fire main, 3=fire right
    action_text = ""
    if action == 0:
        action_text = "nothing"
    elif action == 1:
        action_text = "left engine"
    elif action == 2:
        action_text = "fire main engine"
    elif action == 3:
        action_text = "fire right engine"
    else:
        action_text = "unknown action"
    print(f"Action taken by the model: {action_text}")
    # vector with (x, y, vx, vy, angle, angular_v, left_leg, right_leg)
    print("Observation:", observation)
    print("Reward received:", reward)
    
    # Check if the episode has ended
    if terminated or truncated:
        print("Episode finished, resetting environment.")
        observation, info = env.reset()

env.close()