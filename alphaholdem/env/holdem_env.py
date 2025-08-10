# env/holdem_env.py
import random

class SimpleHoldemEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.deck = [f"{s}{r}" for s in "SHDC" for r in "23456789TJQKA"]
        random.shuffle(self.deck)
        self.hole_cards = [self.deck.pop(), self.deck.pop()]
        self.opp_hole = [self.deck.pop(), self.deck.pop()]
        self.board = []
        self.pot = 2  # small and big blind
        self.bets = [1, 2]  # blinds
        self.current_player = 0  # 0: agent, 1: opponent
        self.phase = 0  # 0: pre-flop, 1: flop, 2: turn, 3: river
        self.done = False
        self.action_seq = []
        return self.get_state()

    def get_state(self):
        return {
            'hole_cards': self.hole_cards,
            'community_cards': self.board.copy(),
            'action_seq': self.action_seq.copy(),
        }

    def step(self, action):
        # Simplified: action in [0,1,2] = fold, call, raise
        # In real: action encodes bet size (9 bins)
        if action == 0:  # fold
            reward = -self.bets[0]
            self.done = True
        elif action == 1:  # call
            reward = 0
            self.pot += self.bets[1] - self.bets[0]
            self.bets[0] = self.bets[1]
        elif action == 2:  # raise
            reward = 0
            self.bets[0] += 10
            self.pot += 10
        # Opponent acts randomly
        opp_action = random.choice([1, 2])  # never folds
        self.action_seq.append((action, opp_action, 2, [1,1,1]))

        # Move to next phase
        if self.phase == 0:
            self.board.extend([self.deck.pop() for _ in range(3)])
        elif self.phase == 1:
            self.board.append(self.deck.pop())
        elif self.phase == 2:
            self.board.append(self.deck.pop())
        self.phase += 1

        if self.phase > 3:
            self.done = True
            reward = self._evaluate_winner() * self.pot

        return self.get_state(), reward, self.done, {}

    def _evaluate_winner(self):
        # Dummy: 50% win
        return 1 if random.random() > 0.5 else -1