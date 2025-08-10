# agents/agent.py
import torch
import torch.nn.functional as F
from models.alpha_holdem_net import AlphaHoldemNet
from utils.state_representation import encode_cards, encode_actions

class AlphaHoldemAgent:
    def __init__(self, nb_actions=9, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = AlphaHoldemNet(nb_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    @torch.no_grad()
    def act(self, hole_cards, community_cards, action_seq):
        card_tensor = torch.FloatTensor(encode_cards(hole_cards, community_cards)).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(encode_actions(action_seq)).unsqueeze(0).to(self.device)
        policy, value = self.model(card_tensor, action_tensor)

        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Safe extraction of scalar values
        return action.item(), log_prob.item(), value.squeeze().item()


    def update(self, states, actions, log_probs, returns, advantages, clip_epsilon=0.2, delta1=3.0, delta2=20000, delta3=20000):
        card_tensors = torch.stack([s[0] for s in states]).to(self.device)
        action_tensors = torch.stack([s[1] for s in states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        policy, value = self.model(card_tensors, action_tensors)
        dist = torch.distributions.Categorical(policy)
        new_log_probs = dist.log_prob(actions)

        # Trinal-Clip PPO: Policy Loss
        ratio = (new_log_probs - log_probs).exp()
        clip_frac = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        if advantages.mean() < 0:
            clip_frac = torch.clamp(ratio, 1 - clip_epsilon, delta1)  # upper bound
        policy_loss = -torch.min(ratio * advantages, clip_frac * advantages).mean()

        # Value Loss with clipping
        clipped_value = torch.clamp(value.squeeze(), -delta2, delta3)
        value_loss = F.mse_loss(clipped_value, returns)

        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
