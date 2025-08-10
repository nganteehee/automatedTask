# train.py
import torch
from agents.agent import AlphaHoldemAgent
from buffer.replay_buffer import ReplayBuffer
from env.holdem_env import SimpleHoldemEnv
from utils.state_representation import encode_cards, encode_actions
import numpy as np

def compute_gae(rewards, values, next_value, gamma=0.999, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * tau * gae
        returns.insert(0, gae + values[i])
    return returns

def train():
    env = SimpleHoldemEnv()
    agent = AlphaHoldemAgent()
    buffer = ReplayBuffer()

    best_elo = 0
    historical_agents = []

    for episode in range(1000):
        state = env.reset()
        done = False
        states, actions, log_probs, rewards, values = [], [], [], [], []

        while not done:
            card_tensor = torch.FloatTensor(encode_cards(state['hole_cards'], state['community_cards']))
            action_tensor = torch.FloatTensor(encode_actions(state['action_seq']))
            action, log_prob, value = agent.act(state['hole_cards'], state['community_cards'], state['action_seq'])

            next_state, reward, done, _ = env.step(action)

            states.append((card_tensor, action_tensor))
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            state = next_state

        # Compute returns and advantages
        returns = compute_gae(rewards, values, 0)
        advantages = [ret - val for ret, val in zip(returns, values)]

        # Update agent
        batch_size = len(rewards)
        for _ in range(3):  # PPO epochs
            loss = agent.update(states, actions, log_probs, returns, advantages)
        print(f"Episode {episode}, Loss: {loss:.4f}, Reward: {sum(rewards):.2f}")

        # K-Best Self-Play: save model if improved
        if sum(rewards) > best_elo:
            best_elo = sum(rewards)
            historical_agents.append((agent.model.state_dict(), best_elo))
            historical_agents = sorted(historical_agents, key=lambda x: -x[1])[:5]  # keep top 5

    torch.save(historical_agents[0][0], "alphaholdem_best.pth")
    print("Training complete.")

if __name__ == "__main__":
    train()
