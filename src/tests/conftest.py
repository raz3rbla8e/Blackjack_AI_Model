import pytest
from game.deck import Deck
from game.hand import Hand
from agents.q_learning_agent import QLearningAgent

@pytest.fixture
def test_deck():
    """Fixture that provides a fresh deck for testing."""
    return Deck()

@pytest.fixture
def test_hand():
    """Fixture that provides an empty hand for testing."""
    return Hand()

@pytest.fixture
def q_learning_agent():
    """Fixture that provides a QLearningAgent instance for testing."""
    return QLearningAgent()
