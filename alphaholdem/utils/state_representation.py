# utils/state_representation.py
import numpy as np

def card_to_index(suit, rank):
    """
    Convert a card's suit and rank to (suit_idx, rank_idx)
    Suit: 'S', 'H', 'D', 'C' → 0,1,2,3
    Rank: '2'-'9', 'T', 'J', 'Q', 'K', 'A' → 0-12
    """
    suit = suit.upper()
    rank = rank.upper()
    
    suit_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
    rank_map = {str(r): r - 2 for r in range(2, 10)}  # '2'->0, ..., '9'->7
    rank_map.update({
        'T': 8,  # Ten
        'J': 9,  # Jack
        'Q': 10, # Queen
        'K': 11, # King
        'A': 12  # Ace
    })
    
    if suit not in suit_map:
        raise ValueError(f"Invalid suit: {suit}. Must be one of S, H, D, C.")
    if rank not in rank_map:
        raise ValueError(f"Invalid rank: {rank}. Must be 2-9, T, J, Q, K, A.")
        
    return suit_map[suit], rank_map[rank]


def encode_cards(hole_cards=None, community_cards=None):
    """
    Encode card information into a 6x4x13 tensor.
    Channels:
        0: Player's hole cards
        1: Flop (first 3 community cards)
        2: Turn (4th community card)
        3: River (5th community card)
        4: All public cards (flop + turn + river)
        5: All cards (hole + public)

    Each card is represented as a 4x13 binary matrix (suits x ranks).
    """
    tensor = np.zeros((6, 4, 13), dtype=np.float32)

    def set_card(channel, card_str):
        if len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")
        rank_char = card_str[0]
        suit_char = card_str[1]
        s, r = card_to_index(suit_char, rank_char)
        channel[s, r] = 1.0

    # Encode hole cards (your two private cards)
    if hole_cards:
        for card in hole_cards:
            set_card(tensor[0], card)

    # Encode community cards
    if community_cards:
        for i, card in enumerate(community_cards):
            set_card(tensor[4], card)  # public cards
            set_card(tensor[5], card)  # all cards (hole + public)
            if i < 3:
                set_card(tensor[1], card)  # flop
            elif i == 3:
                set_card(tensor[2], card)  # turn
            elif i == 4:
                set_card(tensor[3], card)  # river

    return tensor


def encode_actions(action_sequence, nb=9):
    """
    Encode action history into a 24x4xnb tensor.
    
    Channels (24 = 4 rounds × 6 max actions):
        Each channel (time step) has 4 rows:
            0: Player 1's action (one-hot or index)
            1: Player 2's action (one-hot or index)
            2: Bet size index (0-8, matching 9 bins)
            3: Legal action mask (for training)

    nb: number of betting options (default 9 as in paper)

    Input action_sequence: list of tuples like
        (p1_action_idx, p2_action_idx, bet_size_idx, legal_mask_list)
        e.g., (2, 1, 4, [1,1,1,1,1,0,0,0,1])
    """
    tensor = np.zeros((24, 4, nb), dtype=np.float32)

    for t, action in enumerate(action_sequence):
        if t >= 24:
            break
        p1_act, p2_act, bet_size_idx, legal_mask = action

        # Clamp indices
        p1_act = min(max(p1_act, 0), nb - 1)
        p2_act = min(max(p2_act, 0), nb - 1)
        bet_size_idx = min(max(bet_size_idx, 0), nb - 1)

        tensor[t, 0, p1_act] = 1.0
        tensor[t, 1, p2_act] = 1.0
        tensor[t, 2, bet_size_idx] = 1.0
        tensor[t, 3, :] = legal_mask  # all 1s if all legal

    return tensor


# --- Example Usage (for debugging) ---
if __name__ == "__main__":
    # Test card encoding
    hole = ['As', 'Tc']  # Ace of Spades, Ten of Clubs
    flop = ['7h', '8d', 'Ks']
    turn = ['Qc']
    river = ['Jh']
    board = flop + turn + river

    card_tensor = encode_cards(hole_cards=hole, community_cards=board)
    print("Card tensor shape:", card_tensor.shape)  # Should be (6, 4, 13)
    print("Hole cards (As, Tc) encoded in channel 0:")
    print(card_tensor[0])  # Should have 1 at (S, A) and (C, T)

    # Test action encoding
    # Simulate: P1 raises (idx=2), P2 calls (idx=1), bet size=3, all actions legal
    actions = [
        (2, 1, 3, [1]*9),
        (1, 0, 1, [1,1,1,0,0,0,0,0,0]),  # call, fold, small bet
    ]
    action_tensor = encode_actions(actions, nb=9)
    print("Action tensor shape:", action_tensor.shape)  # Should be (24, 4, 9)
    print("First action - P1 action:", np.argmax(action_tensor[0, 0, :]))
    print("First action - P2 action:", np.argmax(action_tensor[0, 1, :]))
    print("First action - Bet size:", np.argmax(action_tensor[0, 2, :]))
    print("First action - Legal mask:", action_tensor[0, 3, :])