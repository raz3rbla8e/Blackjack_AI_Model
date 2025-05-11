from abc import ABC, abstractmethod
from typing import Tuple, Any
from game.hand import Hand
from game.deck import Deck

#abstract agent class
class BaseAgent(ABC):
    @abstractmethod
    def getAction(self, playerHand: Hand, dealerHand: Hand, gameDeck: Deck, split_depth: int = 0) -> int:
        pass
    
    @abstractmethod
    def update_Q(self, state: Tuple, action: int, reward: float, next_state: Tuple, terminal: bool) -> None:
        pass
    
    @abstractmethod
    def getState(self, playerHand: Hand, dealerHand: Hand, gameDeck: Deck) -> Tuple:
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'BaseAgent':
        pass