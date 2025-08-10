import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
import copy
import math
from enum import Enum

# ============================================================================
# POKER GAME ENVIRONMENT
# ============================================================================

class Card:
    SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.suit_idx = self.SUITS.index(suit)
        self.rank_idx = self.RANKS.index(rank)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self):
        return hash((self.suit, self.rank))

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_QUARTER = 3  # bet 1/4 pot
    BET_HALF = 4     # bet 1/2 pot
    BET_POT = 5      # bet pot size
    BET_2POT = 6     # bet 2x pot
    ALL_IN = 7       # all in
    RAISE_QUARTER = 8 # raise 1/4 pot

class GamePhase(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class HunlPokerEnv:
    """Heads-up No-limit Texas Hold'em Environment"""
    
    def __init__(self, stack_size: int = 20000, big_blind: int = 100):
        self.stack_size = stack_size
        self.big_blind = big_blind
        self.small_blind = big_blind // 2
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        self.deck = self._create_deck()
        self.community_cards = []
        self.hole_cards = [[], []]
        self.pot = 0
        self.stacks = [self.stack_size, self.stack_size]
        self.phase = GamePhase.PREFLOP
        self.current_player = 0  # 0 = small blind, 1 = big blind
        self.button = 0  # dealer button position
        self.action_history = []
        self.round_actions = []
        self.bets = [0, 0]
        self.folded = [False, False]
        self.game_over = False
        self.winner = None
        
        # Deal hole cards
        for i in range(2):
            for player in range(2):
                self.hole_cards[player].append(self.deck.pop())
        
        # Post blinds
        self._post_blinds()
        return self._get_state()
    
    def _create_deck(self):
        """Create and shuffle a deck of cards"""
        deck = []
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                deck.append(Card(suit, rank))
        random.shuffle(deck)
        return deck
    
    def _post_blinds(self):
        """Post small and big blinds"""
        self.stacks[0] -= self.small_blind
        self.stacks[1] -= self.big_blind
        self.bets[0] = self.small_blind
        self.bets[1] = self.big_blind
        self.pot = self.small_blind + self.big_blind
    
    def _get_legal_actions(self) -> List[int]:
        """Get legal actions for current player"""
        legal_actions = []
        opponent = 1 - self.current_player
        
        # Can always fold
        legal_actions.append(Action.FOLD.value)
        
        # Check/Call logic
        if self.bets[self.current_player] == self.bets[opponent]:
            legal_actions.append(Action.CHECK.value)
        else:
            if self.stacks[self.current_player] >= (self.bets[opponent] - self.bets[self.current_player]):
                legal_actions.append(Action.CALL.value)
        
        # Betting/Raising logic
        if self.stacks[self.current_player] > 0:
            pot_size = self.pot
            min_bet = max(self.big_blind, self.bets[opponent] - self.bets[self.current_player])
            
            # Different bet sizes
            for bet_action in [Action.BET_QUARTER, Action.BET_HALF, Action.BET_POT, 
                              Action.BET_2POT, Action.ALL_IN]:
                if bet_action == Action.BET_QUARTER:
                    bet_size = pot_size // 4
                elif bet_action == Action.BET_HALF:
                    bet_size = pot_size // 2
                elif bet_action == Action.BET_POT:
                    bet_size = pot_size
                elif bet_action == Action.BET_2POT:
                    bet_size = pot_size * 2
                else:  # ALL_IN
                    bet_size = self.stacks[self.current_player]
                
                if bet_size >= min_bet and bet_size <= self.stacks[self.current_player]:
                    legal_actions.append(bet_action.value)
        
        return legal_actions
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute an action and return next state"""
        if self.game_over:
            return self._get_state(), 0, True, {}
        
        legal_actions = self._get_legal_actions()
        if action not in legal_actions:
            # Invalid action - default to fold
            action = Action.FOLD.value
        
        reward = self._execute_action(action)
        self.action_history.append((self.current_player, action))
        self.round_actions.append((self.current_player, action))
        
        # Check if round/game is over
        done = self._check_game_over()
        
        if not done:
            self._advance_game()
        
        return self._get_state(), reward, done, {"legal_actions": self._get_legal_actions()}
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return immediate reward"""
        action_enum = Action(action)
        opponent = 1 - self.current_player
        
        if action_enum == Action.FOLD:
            self.folded[self.current_player] = True
            self.game_over = True
            self.winner = opponent
            return -self.bets[self.current_player]  # Loss
        
        elif action_enum == Action.CHECK:
            pass  # No bet change
        
        elif action_enum == Action.CALL:
            call_amount = self.bets[opponent] - self.bets[self.current_player]
            self.stacks[self.current_player] -= call_amount
            self.bets[self.current_player] += call_amount
            self.pot += call_amount
        
        else:  # Betting/Raising actions
            bet_size = self._calculate_bet_size(action_enum)
            total_bet = self.bets[self.current_player] + bet_size
            
            self.stacks[self.current_player] -= bet_size
            self.bets[self.current_player] = total_bet
            self.pot += bet_size
        
        return 0  # No immediate reward unless game ends
    
    def _calculate_bet_size(self, action: Action) -> int:
        """Calculate bet size for betting actions"""
        pot_size = self.pot
        
        if action == Action.BET_QUARTER:
            return pot_size // 4
        elif action == Action.BET_HALF:
            return pot_size // 2
        elif action == Action.BET_POT:
            return pot_size
        elif action == Action.BET_2POT:
            return pot_size * 2
        elif action == Action.ALL_IN:
            return self.stacks[self.current_player]
        
        return 0
    
    def _check_game_over(self) -> bool:
        """Check if the game is over"""
        if any(self.folded):
            return True
        
        # Check if betting round is complete
        if self.bets[0] == self.bets[1] and len(self.round_actions) >= 2:
            # Move to next phase or showdown
            if self.phase == GamePhase.RIVER:
                self.game_over = True
                self._determine_winner()
                return True
            else:
                self._next_phase()
        
        return False
    
    def _next_phase(self):
        """Move to next phase of the game"""
        # Deal community cards based on phase
        if self.phase == GamePhase.PREFLOP:
            # Deal flop (3 cards)
            for _ in range(3):
                self.community_cards.append(self.deck.pop())
            self.phase = GamePhase.FLOP
        elif self.phase == GamePhase.FLOP:
            # Deal turn (1 card)
            self.community_cards.append(self.deck.pop())
            self.phase = GamePhase.TURN
        elif self.phase == GamePhase.TURN:
            # Deal river (1 card)
            self.community_cards.append(self.deck.pop())
            self.phase = GamePhase.RIVER
        
        # Reset betting for new round
        self.bets = [0, 0]
        self.round_actions = []
        self.current_player = 1 - self.button  # Out of position acts first
    
    def _determine_winner(self):
        """Determine winner at showdown"""
        # Simplified hand evaluation - in practice, use proper poker hand evaluation
        hand1 = self._evaluate_hand(0)
        hand2 = self._evaluate_hand(1)
        
        if hand1 > hand2:
            self.winner = 0
        elif hand2 > hand1:
            self.winner = 1
        else:
            self.winner = None  # Tie
    
    def _evaluate_hand(self, player: int) -> int:
        """Simplified hand evaluation (returns higher values for better hands)"""
        # This is a simplified version - real implementation would need proper poker hand evaluation
        cards = self.hole_cards[player] + self.community_cards
        ranks = [card.rank_idx for card in cards]
        return max(ranks)  # Simplified: just return highest card
    
    def _advance_game(self):
        """Advance to next player"""
        self.current_player = 1 - self.current_player
    
    def _get_state(self) -> Dict:
        """Get current game state"""
        return {
            'hole_cards': self.hole_cards,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'stacks': self.stacks,
            'phase': self.phase,
            'current_player': self.current_player,
            'action_history': self.action_history,
            'bets': self.bets,
            'legal_actions': self._get_legal_actions()
        }

# ============================================================================
# STATE REPRESENTATION
# ============================================================================

class StateRepresentation:
    """Convert game state to tensor representation as described in paper"""
    
    def __init__(self, num_bet_actions: int = 9):
        self.num_bet_actions = num_bet_actions
        self.card_channels = 6  # hole, flop, turn, river, all public, all cards
        self.action_channels = 24  # 6 actions per round * 4 rounds
    
    def encode_state(self, state: Dict, player: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode game state into card and action tensors"""
        card_tensor = self._encode_cards(state, player)
        action_tensor = self._encode_actions(state, player)
        
        return card_tensor, action_tensor
    
    def _encode_cards(self, state: Dict, player: int) -> torch.Tensor:
        """Encode card information into 6-channel tensor (4x13 each)"""
        card_tensor = torch.zeros(6, 4, 13)  # 6 channels, 4 suits, 13 ranks
        
        # Channel 0: Player's hole cards
        for card in state['hole_cards'][player]:
            card_tensor[0, card.suit_idx, card.rank_idx] = 1
        
        # Channels 1-4: Community cards by phase
        community = state['community_cards']
        
        if len(community) >= 3:  # Flop
            for i, card in enumerate(community[:3]):
                card_tensor[1, card.suit_idx, card.rank_idx] = 1
        
        if len(community) >= 4:  # Turn
            card = community[3]
            card_tensor[2, card.suit_idx, card.rank_idx] = 1
        
        if len(community) >= 5:  # River
            card = community[4]
            card_tensor[3, card.suit_idx, card.rank_idx] = 1
        
        # Channel 4: All public cards
        for card in community:
            card_tensor[4, card.suit_idx, card.rank_idx] = 1
        
        # Channel 5: All cards (hole + public)
        all_cards = state['hole_cards'][player] + community
        for card in all_cards:
            card_tensor[5, card.suit_idx, card.rank_idx] = 1
        
        return card_tensor
    
    def _encode_actions(self, state: Dict, player: int) -> torch.Tensor:
        """Encode action information into 24-channel tensor"""
        action_tensor = torch.zeros(24, 4, self.num_bet_actions)
        
        # Process action history by rounds
        current_channel = 0
        round_actions = []
        
        for action_player, action in state['action_history']:
            # Each round gets 6 channels max (in practice fewer)
            if current_channel < 24:
                # Row 0: Player 1 actions, Row 1: Player 2 actions
                # Row 2: Sum of actions, Row 3: Legal actions
                row = action_player
                col = min(action, self.num_bet_actions - 1)
                action_tensor[current_channel, row, col] = 1
                
                # Sum row
                action_tensor[current_channel, 2, col] += 1
        
        # Legal actions in current state
        if current_channel < 24:
            for legal_action in state['legal_actions']:
                col = min(legal_action, self.num_bet_actions - 1)
                action_tensor[current_channel, 3, col] = 1
        
        return action_tensor

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class PseudoSiameseNet(nn.Module):
    """Pseudo-Siamese architecture for AlphaHoldem"""
    
    def __init__(self, num_actions: int = 9):
        super().__init__()
        self.num_actions = num_actions
        
        # Card ConvNet branch
        self.card_conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # Action ConvNet branch  
        self.action_conv = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(128 * 4 + 128 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(256, num_actions)
        
        # Value head
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, card_input: torch.Tensor, action_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through pseudo-siamese architecture"""
        # Process card information
        card_features = self.card_conv(card_input)
        card_features = card_features.view(card_features.size(0), -1)
        
        # Process action information
        action_features = self.action_conv(action_input)
        action_features = action_features.view(action_features.size(0), -1)
        
        # Fuse features
        combined = torch.cat([card_features, action_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Output heads
        policy_logits = self.policy_head(fused_features)
        value = self.value_head(fused_features)
        
        return policy_logits, value

# ============================================================================
# TRINAL-CLIP PPO
# ============================================================================

class TrinaClipPPO:
    """Trinal-Clip PPO implementation with policy and value clipping"""
    
    def __init__(self, model: nn.Module, lr: float = 3e-4, 
                 epsilon: float = 0.2, delta1: float = 3.0,
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.epsilon = epsilon
        self.delta1 = delta1
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def compute_loss(self, states_card: torch.Tensor, states_action: torch.Tensor,
                    actions: torch.Tensor, old_log_probs: torch.Tensor,
                    advantages: torch.Tensor, returns: torch.Tensor,
                    delta2: float, delta3: float) -> torch.Tensor:
        """Compute Trinal-Clip PPO loss"""
        
        # Forward pass
        policy_logits, values = self.model(states_card, states_action)
        
        # Current policy distribution
        policy_dist = Categorical(logits=policy_logits)
        log_probs = policy_dist.log_prob(actions)
        entropy = policy_dist.entropy()
        
        # Ratio for PPO
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Trinal-Clip Policy Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        # Apply additional clipping for negative advantages
        mask_neg_adv = advantages < 0
        clipped_ratio = torch.where(
            mask_neg_adv,
            torch.clamp(ratio, min=None, max=self.delta1),
            ratio
        )
        surr3 = clipped_ratio * advantages
        
        policy_loss = -torch.min(torch.min(surr1, surr2), surr3).mean()
        
        # Trinal-Clip Value Loss
        clipped_returns = torch.clamp(returns, -delta2, delta3)
        value_loss = F.mse_loss(values.squeeze(), clipped_returns)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss - 
                     self.entropy_coef * entropy.mean())
        
        return total_loss, policy_loss, value_loss, entropy.mean()
    
    def update(self, batch_data: Dict, delta2: float, delta3: float):
        """Update model parameters"""
        loss, policy_loss, value_loss, entropy = self.compute_loss(
            batch_data['states_card'],
            batch_data['states_action'],
            batch_data['actions'],
            batch_data['old_log_probs'],
            batch_data['advantages'],
            batch_data['returns'],
            delta2, delta3
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

# ============================================================================
# K-BEST SELF-PLAY
# ============================================================================

Experience = namedtuple('Experience', 
                       ['state_card', 'state_action', 'action', 'reward', 
                        'next_state_card', 'next_state_action', 'done', 'log_prob', 'value'])

class KBestSelfPlay:
    """K-Best Self-Play algorithm for model evaluation and generation"""
    
    def __init__(self, k: int = 5):
        self.k = k
        self.agent_pool = deque(maxlen=k)
        self.elo_scores = {}
        self.agent_counter = 0
    
    def add_agent(self, agent_state_dict: Dict, initial_elo: float = 1200):
        """Add a new agent to the pool"""
        agent_id = self.agent_counter
        self.agent_counter += 1
        
        self.agent_pool.append((agent_id, copy.deepcopy(agent_state_dict)))
        self.elo_scores[agent_id] = initial_elo
        
        return agent_id
    
    def get_opponent(self, exclude_id: Optional[int] = None) -> Tuple[int, Dict]:
        """Get opponent agent from pool"""
        available_agents = [(aid, state_dict) for aid, state_dict in self.agent_pool 
                          if aid != exclude_id]
        
        if not available_agents:
            return None, None
        
        # Select opponent based on ELO scores (probabilistic selection)
        agent_ids = [aid for aid, _ in available_agents]
        elo_values = [self.elo_scores[aid] for aid in agent_ids]
        
        # Convert ELO to probabilities
        max_elo = max(elo_values)
        exp_elos = [math.exp((elo - max_elo) / 400) for elo in elo_values]
        total = sum(exp_elos)
        probs = [exp_elo / total for exp_elo in exp_elos]
        
        # Sample opponent
        opponent_idx = np.random.choice(len(available_agents), p=probs)
        return available_agents[opponent_idx]
    
    def update_elo(self, winner_id: int, loser_id: int, k_factor: float = 32):
        """Update ELO scores after a game"""
        winner_elo = self.elo_scores[winner_id]
        loser_elo = self.elo_scores[loser_id]
        
        # Expected scores
        expected_winner = 1 / (1 + 10**((loser_elo - winner_elo) / 400))
        expected_loser = 1 / (1 + 10**((winner_elo - loser_elo) / 400))
        
        # Update ELO
        self.elo_scores[winner_id] = winner_elo + k_factor * (1 - expected_winner)
        self.elo_scores[loser_id] = loser_elo + k_factor * (0 - expected_loser)

# ============================================================================
# TRAINING LOOP
# ============================================================================

class AlphaHoldemTrainer:
    """Main training class for AlphaHoldem"""
    
    def __init__(self, lr: float = 3e-4, gamma: float = 0.999, 
                 gae_lambda: float = 0.95, batch_size: int = 2048):
        
        self.env = HunlPokerEnv()
        self.state_rep = StateRepresentation()
        self.model = PseudoSiameseNet()
        self.ppo = TrinaClipPPO(self.model, lr=lr)
        self.k_best = KBestSelfPlay()
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        
        # Experience buffer
        self.buffer = []
        
        # Add initial agent to pool
        self.current_agent_id = self.k_best.add_agent(self.model.state_dict())
    
    def collect_experience(self, num_games: int = 128) -> List[Experience]:
        """Collect experience through self-play"""
        experiences = []
        
        for _ in range(num_games):
            # Get opponent
            opponent_id, opponent_state = self.k_best.get_opponent(self.current_agent_id)
            
            if opponent_state is not None:
                # Load opponent model
                opponent_model = PseudoSiameseNet()
                opponent_model.load_state_dict(opponent_state)
                opponent_model.eval()
            else:
                opponent_model = self.model
            
            # Play game
            game_experiences = self._play_game(opponent_model)
            experiences.extend(game_experiences)
        
        return experiences
    
    def _play_game(self, opponent_model: nn.Module, opponent_id: Optional[int] = None) -> List[Experience]:
        """Play a single game and collect experiences"""
        state = self.env.reset()
        experiences = []
        
        while not self.env.game_over:
            current_player = state['current_player']
            
            # Choose model based on current player
            if current_player == 0:  # Main agent
                model = self.model
                model.eval()
            else:  # Opponent
                model = opponent_model
                model.eval()
            
            # Encode state
            card_tensor, action_tensor = self.state_rep.encode_state(state, current_player)
            card_tensor = card_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
            
            # Get action
            with torch.no_grad():
                policy_logits, value = model(card_tensor, action_tensor)
                
                # Mask illegal actions
                legal_actions = state['legal_actions']
                action_mask = torch.full_like(policy_logits, float('-inf'))
                for legal_action in legal_actions:
                    action_mask[0, legal_action] = 0
                
                masked_logits = policy_logits + action_mask
                policy_dist = Categorical(logits=masked_logits)
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action.item())
            
            # Store experience (only for main agent)
            if current_player == 0:
                exp = Experience(
                    state_card=card_tensor.squeeze(0),
                    state_action=action_tensor.squeeze(0),
                    action=action,
                    reward=reward,
                    next_state_card=None,  # Will be filled later
                    next_state_action=None,
                    done=done,
                    log_prob=log_prob,
                    value=value.squeeze()
                )
                experiences.append(exp)
            
            state = next_state
            
            if done:
                # Determine winner and update ELO
                if self.env.winner is not None and opponent_id is not None:
                    winner_id = self.current_agent_id if self.env.winner == 0 else opponent_id
                    loser_id = opponent_id if self.env.winner == 0 else self.current_agent_id
                    self.k_best.update_elo(winner_id, loser_id)
                break
        
        return experiences
    
    def compute_gae(self, experiences: List[Experience]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        values = torch.stack([exp.value for exp in experiences])
        dones = torch.tensor([exp.done for exp in experiences], dtype=torch.float32)
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(experiences))):
            if t == len(experiences) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, experiences: List[Experience]) -> Dict:
        """Perform one training step"""
        if len(experiences) == 0:
            return {}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(experiences)
        
        # Prepare batch data
        states_card = torch.stack([exp.state_card for exp in experiences])
        states_action = torch.stack([exp.state_action for exp in experiences])
        actions = torch.stack([exp.action for exp in experiences])
        old_log_probs = torch.stack([exp.log_prob for exp in experiences])
        
        # Calculate dynamic delta2 and delta3 based on chip stacks
        max_chips = self.env.stack_size
        delta2 = float(max_chips)
        delta3 = float(max_chips)
        
        batch_data = {
            'states_card': states_card,
            'states_action': states_action,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns
        }
        
        # Update model
        self.model.train()
        metrics = self.ppo.update(batch_data, delta2, delta3)
        
        return metrics
    
    def train(self, num_iterations: int = 50000, save_freq: int = 1000):
        """Main training loop"""
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Collect experience
            experiences = self.collect_experience(num_games=128)
            
            # Train model
            metrics = self.train_step(experiences)
            
            # Print metrics
            if metrics:
                print(f"Loss: {metrics['total_loss']:.4f}, "
                      f"Policy Loss: {metrics['policy_loss']:.4f}, "
                      f"Value Loss: {metrics['value_loss']:.4f}, "
                      f"Entropy: {metrics['entropy']:.4f}")
            
            # Update agent pool periodically
            if (iteration + 1) % 100 == 0:
                self.current_agent_id = self.k_best.add_agent(
                    self.model.state_dict(), 
                    initial_elo=1200
                )
                print(f"Added new agent to pool. Current ELO scores: {self.k_best.elo_scores}")
            
            # Save model periodically
            if (iteration + 1) % save_freq == 0:
                torch.save(self.model.state_dict(), f'alphaholdem_iter_{iteration + 1}.pth')
                print(f"Model saved at iteration {iteration + 1}")

# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================

class PokerHandEvaluator:
    """Proper poker hand evaluation"""
    
    HAND_RANKINGS = {
        'high_card': 1,
        'pair': 2,
        'two_pair': 3,
        'three_of_a_kind': 4,
        'straight': 5,
        'flush': 6,
        'full_house': 7,
        'four_of_a_kind': 8,
        'straight_flush': 9,
        'royal_flush': 10
    }
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int]]:
        """Evaluate poker hand strength"""
        if len(cards) < 5:
            return 1, [max(card.rank_idx for card in cards)]
        
        # Get all 5-card combinations
        from itertools import combinations
        best_hand = None
        best_rank = 0
        best_tiebreakers = []
        
        for combo in combinations(cards, 5):
            rank, tiebreakers = PokerHandEvaluator._evaluate_5_cards(combo)
            if rank > best_rank or (rank == best_rank and tiebreakers > best_tiebreakers):
                best_hand = combo
                best_rank = rank
                best_tiebreakers = tiebreakers
        
        return best_rank, best_tiebreakers
    
    @staticmethod
    def _evaluate_5_cards(cards: Tuple[Card, ...]) -> Tuple[int, List[int]]:
        """Evaluate exactly 5 cards"""
        ranks = [card.rank_idx for card in cards]
        suits = [card.suit_idx for card in cards]
        
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        is_flush = len(set(suits)) == 1
        is_straight = PokerHandEvaluator._is_straight(ranks)
        
        # Sort ranks by count, then by rank value
        sorted_ranks = sorted(rank_counts.keys(), key=lambda x: (rank_counts[x], x), reverse=True)
        counts = sorted(rank_counts.values(), reverse=True)
        
        # Royal flush
        if is_flush and is_straight and min(ranks) == 8:  # T, J, Q, K, A
            return 10, []
        
        # Straight flush
        if is_flush and is_straight:
            return 9, [max(ranks)]
        
        # Four of a kind
        if counts == [4, 1]:
            return 8, [sorted_ranks[0], sorted_ranks[1]]
        
        # Full house
        if counts == [3, 2]:
            return 7, [sorted_ranks[0], sorted_ranks[1]]
        
        # Flush
        if is_flush:
            return 6, sorted(ranks, reverse=True)
        
        # Straight
        if is_straight:
            return 5, [max(ranks)]
        
        # Three of a kind
        if counts == [3, 1, 1]:
            return 4, [sorted_ranks[0]] + sorted(sorted_ranks[1:], reverse=True)
        
        # Two pair
        if counts == [2, 2, 1]:
            pairs = sorted([r for r in sorted_ranks[:2]], reverse=True)
            return 3, pairs + [sorted_ranks[2]]
        
        # One pair
        if counts == [2, 1, 1, 1]:
            return 2, [sorted_ranks[0]] + sorted(sorted_ranks[1:], reverse=True)
        
        # High card
        return 1, sorted(ranks, reverse=True)
    
    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        sorted_ranks = sorted(set(ranks))
        if len(sorted_ranks) < 5:
            return False
        
        # Check for regular straight
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i + 4] - sorted_ranks[i] == 4:
                return True
        
        # Check for A-2-3-4-5 straight (wheel)
        if sorted_ranks == [0, 1, 2, 3, 12]:  # 2,3,4,5,A
            return True
        
        return False

class AlphaHoldemEvaluator:
    """Evaluation class for AlphaHoldem against other agents"""
    
    def __init__(self, model_path: str):
        self.model = PseudoSiameseNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.state_rep = StateRepresentation()
    
    def play_against_agent(self, opponent_agent, num_games: int = 1000) -> Dict:
        """Play against another agent and return statistics"""
        wins = 0
        total_profit = 0
        game_results = []
        
        for game_num in range(num_games):
            env = HunlPokerEnv()
            state = env.reset()
            
            # Randomly assign positions
            alphaholdem_position = random.choice([0, 1])
            
            while not env.game_over:
                current_player = state['current_player']
                
                if current_player == alphaholdem_position:
                    action = self._get_alphaholdem_action(state, current_player)
                else:
                    action = opponent_agent.get_action(state, current_player)
                
                state, reward, done, info = env.step(action)
                
                if done:
                    if env.winner == alphaholdem_position:
                        wins += 1
                        profit = env.pot - env.stacks[1 - alphaholdem_position]
                    elif env.winner == (1 - alphaholdem_position):
                        profit = -env.stacks[alphaholdem_position]
                    else:  # Tie
                        profit = 0
                    
                    total_profit += profit
                    game_results.append(profit)
                    break
        
        win_rate = wins / num_games
        avg_profit = total_profit / num_games
        
        # Calculate mbb/h (milli-big-blinds per hand)
        mbb_per_hand = (avg_profit / env.big_blind) * 1000
        
        return {
            'games_played': num_games,
            'wins': wins,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_game': avg_profit,
            'mbb_per_hand': mbb_per_hand,
            'game_results': game_results
        }
    
    def _get_alphaholdem_action(self, state: Dict, player: int) -> int:
        """Get action from AlphaHoldem model"""
        card_tensor, action_tensor = self.state_rep.encode_state(state, player)
        card_tensor = card_tensor.unsqueeze(0)
        action_tensor = action_tensor.unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, _ = self.model(card_tensor, action_tensor)
            
            # Mask illegal actions
            legal_actions = state['legal_actions']
            action_mask = torch.full_like(policy_logits, float('-inf'))
            for legal_action in legal_actions:
                if legal_action < policy_logits.size(1):
                    action_mask[0, legal_action] = 0
            
            masked_logits = policy_logits + action_mask
            action = torch.argmax(masked_logits, dim=1).item()
            
            # Ensure action is legal
            if action not in legal_actions:
                action = random.choice(legal_actions)
        
        return action

# ============================================================================
# BASELINE AGENTS FOR COMPARISON
# ============================================================================

class RandomAgent:
    """Random agent that chooses legal actions randomly"""
    
    def get_action(self, state: Dict, player: int) -> int:
        legal_actions = state['legal_actions']
        return random.choice(legal_actions)

class TightAggressiveAgent:
    """Tight-aggressive baseline agent"""
    
    def __init__(self):
        # Simplified hand strength evaluation
        self.strong_hands = [
            ('A', 'A'), ('K', 'K'), ('Q', 'Q'), ('J', 'J'), ('T', 'T'),
            ('A', 'K'), ('A', 'Q'), ('A', 'J'), ('K', 'Q'), ('K', 'J')
        ]
        
    def get_action(self, state: Dict, player: int) -> int:
        legal_actions = state['legal_actions']
        hole_cards = state['hole_cards'][player]
        
        # Simple hand strength evaluation
        if len(hole_cards) >= 2:
            hand_strength = self._evaluate_hole_cards(hole_cards)
        else:
            hand_strength = 0.0
        
        # Decision making based on hand strength
        if hand_strength > 0.8:  # Very strong hand
            # Be aggressive - bet/raise if possible
            aggressive_actions = [a for a in legal_actions 
                                if a in [Action.BET_POT.value, Action.BET_HALF.value, Action.CALL.value]]
            if aggressive_actions:
                return random.choice(aggressive_actions)
        
        elif hand_strength > 0.5:  # Medium hand
            # Play cautiously - call or check
            if Action.CALL.value in legal_actions:
                return Action.CALL.value
            elif Action.CHECK.value in legal_actions:
                return Action.CHECK.value
        
        # Weak hand - fold or check
        if Action.CHECK.value in legal_actions:
            return Action.CHECK.value
        else:
            return Action.FOLD.value
    
    def _evaluate_hole_cards(self, hole_cards: List[Card]) -> float:
        """Simple hole card strength evaluation"""
        if len(hole_cards) < 2:
            return 0.0
        
        card1, card2 = hole_cards[:2]
        
        # Check for premium hands
        hand = tuple(sorted([card1.rank, card2.rank], reverse=True))
        if hand in [('A', 'A'), ('K', 'K'), ('Q', 'Q'), ('J', 'J')]:
            return 1.0
        
        if hand in [('A', 'K'), ('A', 'Q'), ('A', 'J')]:
            return 0.9
        
        if hand[0] == hand[1]:  # Pocket pair
            return 0.7
        
        if card1.suit == card2.suit:  # Suited
            return 0.6
        
        # High cards
        high_ranks = ['A', 'K', 'Q', 'J', 'T']
        if card1.rank in high_ranks or card2.rank in high_ranks:
            return 0.5
        
        return 0.3

# ============================================================================
# MAIN EXECUTION AND DEMO
# ============================================================================

def main():
    """Main function to demonstrate AlphaHoldem training and evaluation"""
    print("AlphaHoldem Implementation")
    print("=" * 50)
    
    # Initialize trainer
    print("Initializing AlphaHoldem trainer...")
    trainer = AlphaHoldemTrainer()
    
    # Demo: Short training run
    print("\nStarting training (demo with 1000 iterations)...")
    try:
        trainer.train(num_iterations=1000, save_freq=5)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    # Save the model
    model_path = 'alphaholdem_demo.pth'
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluation against baseline agents
    print("\nEvaluating against baseline agents...")
    evaluator = AlphaHoldemEvaluator(model_path)
    
    # Test against random agent
    print("Playing against Random Agent (100 games)...")
    random_agent = RandomAgent()
    results_random = evaluator.play_against_agent(random_agent, num_games=100)
    
    print(f"Results vs Random Agent:")
    print(f"  Win Rate: {results_random['win_rate']:.2%}")
    print(f"  mbb/h: {results_random['mbb_per_hand']:.2f}")
    
    # Test against tight-aggressive agent
    print("\nPlaying against Tight-Aggressive Agent (100 games)...")
    ta_agent = TightAggressiveAgent()
    results_ta = evaluator.play_against_agent(ta_agent, num_games=100)
    
    print(f"Results vs Tight-Aggressive Agent:")
    print(f"  Win Rate: {results_ta['win_rate']:.2%}")
    print(f"  mbb/h: {results_ta['mbb_per_hand']:.2f}")
    
    # Demo single game
    print("\nDemo: Single game visualization...")
    demo_game(trainer.model)

def demo_game(model: nn.Module):
    """Demonstrate a single game with action logging"""
    env = HunlPokerEnv()
    state_rep = StateRepresentation()
    model.eval()
    
    state = env.reset()
    print(f"\nGame started!")
    print(f"Player 0 hole cards: {[str(card) for card in state['hole_cards'][0]]}")
    print(f"Player 1 hole cards: {[str(card) for card in state['hole_cards'][1]]}")
    
    action_names = {
        0: "FOLD", 1: "CHECK", 2: "CALL", 3: "BET_1/4", 4: "BET_1/2",
        5: "BET_POT", 6: "BET_2POT", 7: "ALL_IN", 8: "RAISE_1/4"
    }
    
    turn = 0
    while not env.game_over and turn < 20:  # Limit turns to prevent infinite loops
        current_player = state['current_player']
        legal_actions = state['legal_actions']
        
        print(f"\nTurn {turn + 1}: Player {current_player} to act")
        print(f"Phase: {state['phase'].name}")
        print(f"Community cards: {[str(card) for card in state['community_cards']]}")
        print(f"Pot: {state['pot']}, Stacks: {state['stacks']}")
        print(f"Legal actions: {[action_names.get(a, str(a)) for a in legal_actions]}")
        
        # Get action from model
        card_tensor, action_tensor = state_rep.encode_state(state, current_player)
        card_tensor = card_tensor.unsqueeze(0)
        action_tensor = action_tensor.unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = model(card_tensor, action_tensor)
            
            # Mask illegal actions
            action_mask = torch.full_like(policy_logits, float('-inf'))
            for legal_action in legal_actions:
                if legal_action < policy_logits.size(1):
                    action_mask[0, legal_action] = 0
            
            masked_logits = policy_logits + action_mask
            action_probs = F.softmax(masked_logits, dim=1)
            action = torch.argmax(masked_logits, dim=1).item()
            
            if action not in legal_actions:
                action = random.choice(legal_actions)
        
        print(f"Player {current_player} chooses: {action_names.get(action, str(action))}")
        print(f"Value estimate: {value.item():.3f}")
        
        state, reward, done, info = env.step(action)
        turn += 1
        
        if done:
            print(f"\nGame over!")
            if env.winner is not None:
                print(f"Winner: Player {env.winner}")
            else:
                print("Game tied")
            break

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()