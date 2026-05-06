import torch, os
from flappy_env import FlappyEnv
from dqn_model import DQN

MODELS_DIR = "./models"
results = []

for file in os.listdir(MODELS_DIR):
    if not file.endswith(".pth"): continue

    model = DQN(8, 256, 2)
    model.load_state_dict(torch.load(f"{MODELS_DIR}/{file}", map_location="cpu"))
    model.eval()

    env = FlappyEnv()
    state = env.reset()
    total_reward = 0

    while True:
        action = model(torch.tensor(state).float().unsqueeze(0)).argmax().item()
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward
        if done: break

    results.append((file, total_reward))

for r in sorted(results, key=lambda x: -x[1]):
    print(r)