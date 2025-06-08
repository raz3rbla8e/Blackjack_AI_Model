"""
Tests for the QLearningAgent class.
"""
import pytest
import numpy as np
from game.card import Card
from game.hand import Hand
from game.deck import Deck
from agents.q_learning_agent import QLearningAgent

def test_initialization():
    """Test that the QLearningAgent initializes with correct default values."""
    agent = QLearningAgent()
    assert agent.alpha == 0.1
    assert agent.gamma == 0.9
    assert agent.epsilon == 0.1
    assert agent.min_epsilon == 0.01
    assert agent.batch_size == 64

def test_get_hand_value():
    """Test the getHandValue method with various card combinations."""
    agent = QLearningAgent()
    
    # Test with no aces
    cards = [Card('H', 10), Card('D', 7)]  # 10 + 7 = 17
    assert agent.getHandValue(cards) == 17
    
    # Test with ace counted as 11
    cards = [Card('H', 1), Card('D', 9)]  # 11 + 9 = 20
    assert agent.getHandValue(cards) == 20
    
    # Test with ace counted as 1
    cards = [Card('H', 1), Card('D', 9), Card('C', 5)]  # 1 + 9 + 5 = 15
    assert agent.getHandValue(cards) == 15
    
    # Test with multiple aces
    cards = [Card('H', 1), Card('D', 1), Card('C', 9)]  # 1 + 11 + 9 = 21
    assert agent.getHandValue(cards) == 21

def test_get_state():
    """Test the getState method returns correct state representation."""
    agent = QLearningAgent()
    deck = Deck()
    
    # Create test hands
    player_hand = Hand()
    dealer_hand = Hand()
    
    # Add specific cards for testing
    player_hand.addCard(Card('H', 10))  # 10
    player_hand.addCard(Card('D', 1))   # Ace
    
    dealer_hand.addCard(Card('C', 5))  # Dealer's upcard: 5
    
    # Test state
    state = agent.getState(player_hand, dealer_hand, deck)
    
    # Unpack state
    player_value, has_usable_ace, dealer_upcard, count_bucket, is_pair, pair_value = state
    
    # Assertions
    assert player_value == 21  # 10 + 11 (ace)
    # has_usable_ace should be False for this hand because the ace is already counted as 11
    assert has_usable_ace is False
    assert dealer_upcard == 5
    # count_bucket is a numpy float64 from np.sign()
    assert isinstance(count_bucket, np.float64)
    assert is_pair is False
    assert pair_value == 0

def test_available_actions():
    """Test the _get_available_actions method."""
    agent = QLearningAgent()
    hand = Hand()
    
    # Test with empty hand - should still return hit and stand
    assert set(agent._get_available_actions(hand)) == {0, 1}
    
    # Test with splittable hand
    hand.addCard(Card('H', 8))
    hand.addCard(Card('D', 8))
    assert set(agent._get_available_actions(hand)) == {0, 1, 2}  # Hit, Stand, Split
    
    # Test with non-splittable hand
    hand = Hand()
    hand.addCard(Card('H', 8))
    hand.addCard(Card('D', 7))
    assert set(agent._get_available_actions(hand)) == {0, 1}  # Hit, Stand

def test_epsilon_decay():
    """Test that epsilon decays correctly over time."""
    agent = QLearningAgent(epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.9)
    
    # Initial epsilon should be 1.0
    assert agent.epsilon == 1.0
    
    # After one decay
    agent.epsilon = agent.epsilon * agent.epsilon_decay
    assert agent.epsilon == 0.9
    
    # After multiple decays, should not go below min_epsilon
    for _ in range(100):
        agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    
    assert agent.epsilon == agent.min_epsilon

# Add more test cases as needed
