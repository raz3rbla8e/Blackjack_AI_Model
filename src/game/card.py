#Represents a card
class Card:
    def __init__(self, suit: str, rank: int):
        self.suit = suit
        self.rank = rank
        
    def getInfo(self) -> tuple:
        return self.suit, self.rank
    
    def getValue(self) -> int:
        return self.rank
    
    #for card counting with hi-lo strategy (https://wizardofodds.com/games/blackjack/card-counting/high-low/)
    def getCountValue(self) -> int:
        if self.rank in [10, 1]:  # 10s and Aces
            return -1
        elif self.rank in [2, 3, 4, 5, 6]:  # Low cards
            return 1
        else:  # 7, 8, 9
            return 0