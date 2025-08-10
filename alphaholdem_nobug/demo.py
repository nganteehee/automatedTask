import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List

# Import the classes from the main implementation
# (Assuming you have the main code in a file called alphaholdem.py)
from alphaholdem import *

# If you're running this in the same file, just use the existing classes

def run_demo_games():
    """Run demo games with trained AlphaHoldem model"""
    
    print("AlphaHoldem Demo Runner")
    print("=" * 50)
    
    # Load the trained model
    model_path = 'alphaholdem_demo.pth'
    
    try:
        evaluator = AlphaHoldemEvaluator(model_path)
        print(f"✓ Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model file {model_path} not found!")
        print("Please make sure you have trained a model first.")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("\nChoose demo option:")
    print("1. Watch single game (detailed)")
    print("2. Play 100 games vs Random Agent")
    print("3. Play 100 games vs Tight-Aggressive Agent")
    print("4. Quick performance test (10 games each)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "="*60)
        print("SINGLE GAME DEMO - DETAILED VIEW")
        print("="*60)
        demo_detailed_game(evaluator)
        
    elif choice == "2":
        print("\n" + "="*60)
        print("100 GAMES vs RANDOM AGENT")
        print("="*60)
        random_agent = RandomAgent()
        results = evaluator.play_against_agent(random_agent, num_games=100)
        print_results("Random Agent", results)
        
    elif choice == "3":
        print("\n" + "="*60)
        print("100 GAMES vs TIGHT-AGGRESSIVE AGENT")
        print("="*60)
        ta_agent = TightAggressiveAgent()
        results = evaluator.play_against_agent(ta_agent, num_games=100)
        print_results("Tight-Aggressive Agent", results)
        
    elif choice == "4":
        print("\n" + "="*60)
        print("QUICK PERFORMANCE TEST")
        print("="*60)
        
        # Test against multiple agents
        agents = {
            "Random Agent": RandomAgent(),
            "Tight-Aggressive Agent": TightAggressiveAgent()
        }
        
        for agent_name, agent in agents.items():
            print(f"\nTesting against {agent_name} (10 games)...")
            results = evaluator.play_against_agent(agent, num_games=10)
            print(f"  Win Rate: {results['win_rate']:.1%}")
            print(f"  mbb/h: {results['mbb_per_hand']:.1f}")
    
    else:
        print("Invalid choice. Please run again and choose 1-4.")

def demo_detailed_game(evaluator: 'AlphaHoldemEvaluator'):
    """Run a detailed single game demo"""
    
    env = HunlPokerEnv()
    state_rep = StateRepresentation()
    
    # Choose opponent
    print("Choose opponent:")
    print("1. Random Agent")
    print("2. Tight-Aggressive Agent")
    print("3. Self-play (AlphaHoldem vs AlphaHoldem)")
    
    opp_choice = input("Enter choice (1-3): ").strip()
    
    if opp_choice == "1":
        opponent = RandomAgent()
        opp_name = "Random Agent"
    elif opp_choice == "2":
        opponent = TightAggressiveAgent()
        opp_name = "Tight-Aggressive Agent"
    else:
        opponent = AlphaHoldemAgent(evaluator.model, state_rep)
        opp_name = "AlphaHoldem (Self-play)"
    
    # Start game
    state = env.reset()
    
    # Randomly assign positions
    alphaholdem_position = random.choice([0, 1])
    opponent_position = 1 - alphaholdem_position
    
    print(f"\n🃏 GAME SETUP 🃏")
    print(f"AlphaHoldem: Player {alphaholdem_position}")
    print(f"{opp_name}: Player {opponent_position}")
    print(f"Stack sizes: {env.stack_size} chips each")
    print(f"Blinds: {env.small_blind}/{env.big_blind}")
    
    print(f"\n📋 HOLE CARDS:")
    print(f"Player 0: {[str(card) for card in state['hole_cards'][0]]}")
    print(f"Player 1: {[str(card) for card in state['hole_cards'][1]]}")
    
    action_names = {
        0: "FOLD", 1: "CHECK", 2: "CALL", 3: "BET_1/4", 4: "BET_1/2",
        5: "BET_POT", 6: "BET_2POT", 7: "ALL_IN", 8: "RAISE_1/4"
    }
    
    turn = 0
    max_turns = 50  # Prevent infinite loops
    
    print(f"\n🎮 GAME PLAY:")
    print("-" * 40)
    
    while not env.game_over and turn < max_turns:
        current_player = state['current_player']
        legal_actions = state['legal_actions']
        
        # Show game state
        print(f"\n🔄 Turn {turn + 1}")
        print(f"Phase: {state['phase'].name}")
        print(f"Current Player: {current_player}")
        
        if state['community_cards']:
            print(f"Community: {[str(card) for card in state['community_cards']]}")
        
        print(f"Pot: ${state['pot']}")
        print(f"Stacks: P0=${state['stacks'][0]}, P1=${state['stacks'][1]}")
        print(f"Current Bets: P0=${state['bets'][0]}, P1=${state['bets'][1]}")
        
        # Get action
        if current_player == alphaholdem_position:
            print(f"🤖 AlphaHoldem thinking...")
            action = evaluator._get_alphaholdem_action(state, current_player)
            
            # Show AlphaHoldem's decision process
            card_tensor, action_tensor = state_rep.encode_state(state, current_player)
            card_tensor = card_tensor.unsqueeze(0)
            action_tensor = action_tensor.unsqueeze(0)
            
            with torch.no_grad():
                policy_logits, value = evaluator.model(card_tensor, action_tensor)
                
                # Show value estimate and action probabilities
                print(f"   Position Value Estimate: {value.item():.3f}")
                
                # Show top 3 action preferences
                legal_mask = torch.full_like(policy_logits, float('-inf'))
                for legal_action in legal_actions:
                    if legal_action < policy_logits.size(1):
                        legal_mask[0, legal_action] = 0
                
                masked_logits = policy_logits + legal_mask
                action_probs = F.softmax(masked_logits, dim=1)
                
                top_actions = torch.topk(action_probs[0], min(3, len(legal_actions)))
                print(f"   Top Actions:")
                for i, (prob, act_idx) in enumerate(zip(top_actions.values, top_actions.indices)):
                    if act_idx.item() in legal_actions:
                        print(f"     {i+1}. {action_names.get(act_idx.item(), str(act_idx.item()))}: {prob.item():.1%}")
            
            player_name = "🤖 AlphaHoldem"
            
        else:
            print(f"🎯 {opp_name} thinking...")
            action = opponent.get_action(state, current_player)
            player_name = f"🎯 {opp_name}"
        
        print(f"   ➤ {player_name} chooses: {action_names.get(action, str(action))}")
        
        # Execute action
        state, reward, done, info = env.step(action)
        turn += 1
        
        if done:
            break
    
    # Show final result
    print(f"\n🏁 GAME OVER!")
    print("-" * 40)
    
    if env.winner is not None:
        winner_name = "🤖 AlphaHoldem" if env.winner == alphaholdem_position else f"🎯 {opp_name}"
        print(f"🏆 Winner: {winner_name} (Player {env.winner})")
        
        # Show final hands if we reached showdown
        if state['phase'] == GamePhase.RIVER and state['community_cards']:
            print(f"\n🃏 FINAL HANDS:")
            evaluator_poker = PokerHandEvaluator()
            
            for player in [0, 1]:
                all_cards = state['hole_cards'][player] + state['community_cards']
                hand_rank, tiebreakers = evaluator_poker.evaluate_hand(all_cards)
                
                player_name = "🤖 AlphaHoldem" if player == alphaholdem_position else f"🎯 {opp_name}"
                print(f"{player_name}: {[str(card) for card in state['hole_cards'][player]]}")
                print(f"  Hand Strength: {hand_rank}")
        
        pot_won = env.pot
        print(f"💰 Pot won: ${pot_won}")
        
    else:
        print("🤝 Game ended in a tie")
    
    print(f"\n📊 Final Stacks:")
    print(f"Player 0: ${state['stacks'][0]}")
    print(f"Player 1: ${state['stacks'][1]}")

def print_results(opponent_name: str, results: Dict):
    """Print formatted results"""
    print(f"\n📊 RESULTS vs {opponent_name}:")
    print(f"  Games Played: {results['games_played']}")
    print(f"  Wins: {results['wins']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Total Profit: ${results['total_profit']:.0f}")
    print(f"  Average Profit per Game: ${results['avg_profit_per_game']:.1f}")
    print(f"  Performance (mbb/h): {results['mbb_per_hand']:.2f}")
    
    # Performance interpretation
    mbb_h = results['mbb_per_hand']
    if mbb_h > 50:
        print("  🔥 Excellent performance!")
    elif mbb_h > 20:
        print("  ✅ Good performance!")
    elif mbb_h > 0:
        print("  📈 Positive performance")
    elif mbb_h > -20:
        print("  📉 Slightly negative performance")
    else:
        print("  ❌ Poor performance")

class AlphaHoldemAgent:
    """Wrapper to make AlphaHoldem model compatible with agent interface"""
    
    def __init__(self, model, state_rep):
        self.model = model
        self.state_rep = state_rep
    
    def get_action(self, state: Dict, player: int) -> int:
        """Get action using AlphaHoldem model"""
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

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    run_demo_games()
