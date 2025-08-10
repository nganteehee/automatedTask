# models/alpha_holdem_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 13))  # standardize output
        )

    def forward(self, x):
        return self.conv(x)

class AlphaHoldemNet(nn.Module):
    def __init__(self, nb_actions=9):
        super().__init__()
        # Card stream: 6 channels → 64 features
        self.card_net = ConvEncoder(6)
        # Action stream: 24 channels → 64 features
        self.action_net = ConvEncoder(24)

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, nb_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, card_tensor, action_tensor):
        # card_tensor: (B, 6, 4, 13)
        # action_tensor: (B, 24, 4, 9)
        card_feat = self.card_net(card_tensor).view(-1, 64)
        act_feat = self.action_net(action_tensor).view(-1, 64)
        fused = torch.cat([card_feat, act_feat], dim=1)
        x = self.fc(fused)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return F.softmax(policy_logits, dim=-1), value

