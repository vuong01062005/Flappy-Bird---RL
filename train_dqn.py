import torch
import torch.optim as optim
import numpy as np

from flappy_env import FlappyEnv
from dqn_model import DQN, ReplayBuffer, compute_td_loss

EPISODES = 3000
GAMMA = 0.99
LR = 1e-4

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.998

TARGET_UPDATE = 20

if __name__ == "__main__":
    env = FlappyEnv()

    model = DQN(8, 256, 2)
    target_model = DQN(8, 256, 2)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    buffer = ReplayBuffer(50000)

    epsilon = EPS_START

    for ep in range(EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = np.random.randint(0, 2)
            else:
                with torch.no_grad():
                    action = model(torch.tensor(state).float()).argmax().item()

            next_state, reward, done = env.step(action)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > 512:
                loss = compute_td_loss(model, target_model, buffer, 256, GAMMA, optimizer)

            if done:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        print(f"Episode {ep} | Reward = {total_reward:.1f} | ε = {epsilon:.3f}")

        if ep % 200 == 0:
            torch.save(model.state_dict(), f"flappy_dqn_{ep}.pth")