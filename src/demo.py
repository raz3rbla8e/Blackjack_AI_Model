import pygame
import sys
import os
import time
from typing import List, Tuple, Optional
from agents.q_learning_agent import QLearningAgent
from game.blackjack import Blackjack
from game.hand import Hand
from game.deck import Deck
from game.card import Card

pygame.init()

WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
CARD_WIDTH = 100
CARD_HEIGHT = 140
CARD_SPACING = 30
FPS = 60

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
GOLD = (218, 165, 32)
BLUE = (0, 0, 139)

class BlackjackDemo:
    def __init__(self, model_path: str):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Blackjack Q-Learning")
        self.clock = pygame.time.Clock()
        
        self.agent = QLearningAgent.load(model_path)
        
        self.game = Blackjack()
        self.deck = Deck()
        self.dealer_hand = Hand()
        self.player_hands = [Hand()]
        self.current_hand_index = 0
        
        self.game_over = False
        self.hand_in_progress = False
        self.dealer_turn = False
        self.last_decision_time = 0
        self.last_action = None
        self.last_action_time = 0
        self.decision_pending = False
        self.decision_delay = 1.0
        
        self.stats = {
            'hands_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'blackjacks': 0,
            'splits': 0
        }
        
        self.card_images = self._load_card_images()
        self.card_back = pygame.image.load('assets/cards/back.png')
        self.card_back = pygame.transform.scale(self.card_back, (CARD_WIDTH, CARD_HEIGHT))

    def _load_card_images(self) -> dict:
        images = {}
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        
        for suit in suits:
            for rank in ranks:
                key = (suit, rank)
                filename = f'assets/cards/{suit}_{rank}.png'
                if os.path.exists(filename):
                    img = pygame.image.load(filename)
                    img = pygame.transform.scale(img, (CARD_WIDTH, CARD_HEIGHT))
                    images[key] = img
        
        return images

    def _draw_card(self, card: Card, x: int, y: int, hidden: bool = False) -> None:
        if hidden:
            self.screen.blit(self.card_back, (x, y))
        else:
            suit, rank = card.getInfo()
            if (suit, rank) in self.card_images:
                self.screen.blit(self.card_images[(suit, rank)], (x, y))

    def _draw_hand(self, hand: Hand, x: int, y: int, hide_first: bool = False) -> None:
        cards = hand.getHand()
        for i, card in enumerate(cards):
            card_x = x + i * CARD_SPACING
            self._draw_card(card, card_x, y, hidden=(hide_first and i == 0))

    def _draw_text(self, text: str, x: int, y: int, color: Tuple[int, int, int] = WHITE, 
                  size: int = 32, centered: bool = False) -> None:
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, color)
        if centered:
            text_rect = text_surface.get_rect(center=(x, y))
        else:
            text_rect = text_surface.get_rect(topleft=(x, y))
        self.screen.blit(text_surface, text_rect)

    def _draw_stats(self) -> None:
        stats_y = 20
        self._draw_text(f"Hands Played: {self.stats['hands_played']}", 20, stats_y)
        self._draw_text(f"Wins: {self.stats['wins']}", 20, stats_y + 30)
        self._draw_text(f"Losses: {self.stats['losses']}", 20, stats_y + 60)
        self._draw_text(f"Draws: {self.stats['draws']}", 20, stats_y + 90)
        self._draw_text(f"Blackjacks: {self.stats['blackjacks']}", 20, stats_y + 120)
        self._draw_text(f"Splits: {self.stats['splits']}", 20, stats_y + 150)
        
        win_rate = (self.stats['wins'] / self.stats['hands_played'] * 100) if self.stats['hands_played'] > 0 else 0
        self._draw_text(f"Win Rate: {win_rate:.1f}%", 20, stats_y + 180)

    def _draw_decision(self) -> None:
        if self.last_action is not None:
            action_text = {
                0: "STAND",
                1: "HIT",
                2: "SPLIT"
            }.get(self.last_action, "UNKNOWN")
            
            decision_surface = pygame.Surface((300, 60))
            decision_surface.set_alpha(180)
            decision_surface.fill(BLACK)
            self.screen.blit(decision_surface, (WINDOW_WIDTH//2 - 150, 350))
            
            self._draw_text("Agent Decision:", WINDOW_WIDTH//2, 360, 
                          color=GOLD, centered=True)
            self._draw_text(action_text, WINDOW_WIDTH//2, 385, 
                          color=WHITE, size=48, centered=True)

    def can_split(self, hand: Hand) -> bool:
        cards = hand.getHand()
        if len(cards) != 2:
            return False
        return cards[0].getValue() == cards[1].getValue()

    def split_hand(self, hand_index: int) -> None:
        original_hand = self.player_hands[hand_index]
        cards = original_hand.getHand()
        
        self.player_hands[hand_index] = Hand()
        self.player_hands[hand_index].addCard(cards[0])
        self.player_hands[hand_index].addCard(self.deck.draw())
        
        new_hand = Hand()
        new_hand.addCard(cards[1])
        new_hand.addCard(self.deck.draw())
        self.player_hands.insert(hand_index + 1, new_hand)
        
        self.stats['splits'] += 1

    def start_new_hand(self) -> None:
        self.dealer_hand = Hand()
        self.player_hands = [Hand()]
        self.current_hand_index = 0
        self.deck = Deck()
        self.game.deal(self.deck, self.dealer_hand, self.player_hands[0])
        self.hand_in_progress = True
        self.dealer_turn = False
        self.game_over = False
        self.last_action = None
        self.decision_pending = True
        self.last_decision_time = time.time()
        self.stats['hands_played'] += 1

    def update_game_state(self) -> None:
        current_time = time.time()
        
        if not self.hand_in_progress:
            return

        if not self.dealer_turn:
            if self.decision_pending and current_time - self.last_decision_time >= self.decision_delay:
                current_hand = self.player_hands[self.current_hand_index]
                action = self.agent.getAction(current_hand, self.dealer_hand, self.deck) #O(1) is sexy
                self.last_action = action
                self.last_action_time = current_time
                self.decision_pending = False
                
                if action == 1:  # Hit
                    current_hand.addCard(self.deck.draw())
                    if self.game.getHandValue(current_hand) > 21:
                        self.current_hand_index += 1
                        if self.current_hand_index >= len(self.player_hands):
                            self.dealer_turn = True
                        else:
                            self.decision_pending = True
                            self.last_decision_time = current_time
                    else:
                        self.decision_pending = True
                        self.last_decision_time = current_time
                elif action == 2 and self.can_split(current_hand):  # Split
                    self.split_hand(self.current_hand_index)
                    self.decision_pending = True
                    self.last_decision_time = current_time
                else:  # Stand or invalid split
                    self.current_hand_index += 1
                    if self.current_hand_index >= len(self.player_hands):
                        self.dealer_turn = True
                    else:
                        self.decision_pending = True
                        self.last_decision_time = current_time

        elif current_time - self.last_action_time >= self.decision_delay:
            dealer_value = self.game.getHandValue(self.dealer_hand)
            if dealer_value < 17:
                self.dealer_hand.addCard(self.deck.draw())
                self.last_action_time = current_time
            else:
                dealer_value = self.game.getHandValue(self.dealer_hand)
                for hand in self.player_hands:
                    player_value = self.game.getHandValue(hand)
                    
                    if player_value > 21:
                        self.stats['losses'] += 1
                    elif dealer_value > 21:
                        self.stats['wins'] += 1
                    elif player_value > dealer_value:
                        self.stats['wins'] += 1
                    elif player_value < dealer_value:
                        self.stats['losses'] += 1
                    else:
                        self.stats['draws'] += 1
                    
                    if player_value == 21 and len(hand.getHand()) == 2:
                        self.stats['blackjacks'] += 1
                
                self.hand_in_progress = False
                self.game_over = True

    def draw(self) -> None:
        self.screen.fill(GREEN)
        
        self._draw_hand(self.dealer_hand, WINDOW_WIDTH//2 - CARD_WIDTH//2, 100, 
                       hide_first=not self.game_over)
        
        hand_spacing = CARD_HEIGHT + 40
        start_y = 400
        for i, hand in enumerate(self.player_hands):
            is_current = i == self.current_hand_index and not self.dealer_turn
            highlight_color = GOLD if is_current else WHITE
            
            hand_y = start_y + i * hand_spacing
            self._draw_hand(hand, WINDOW_WIDTH//2 - CARD_WIDTH//2, hand_y)
            self._draw_text(f"Hand {i+1}", WINDOW_WIDTH//2, hand_y - 30, 
                          color=highlight_color, centered=True)
            self._draw_text(f"Value: {self.game.getHandValue(hand)}", 
                          WINDOW_WIDTH//2, hand_y + CARD_HEIGHT + 10, 
                          color=highlight_color, centered=True)
        
        self._draw_text("Dealer", WINDOW_WIDTH//2, 50, centered=True)
        
        dealer_value = self.game.getHandValue(self.dealer_hand)
        if self.game_over or self.dealer_turn:
            self._draw_text(f"Value: {dealer_value}", WINDOW_WIDTH//2, 300, centered=True)
        else:
            visible_card = self.dealer_hand.getHand()[1]
            self._draw_text(f"Showing: {visible_card.getValue()}", WINDOW_WIDTH//2, 300, centered=True)
        
        if self.hand_in_progress and not self.game_over:
            self._draw_decision()
        
        self._draw_stats()
        
        self._draw_text("Space: New Hand    Esc: Quit    Up/Down: Adjust Speed", 
                       WINDOW_WIDTH//2, WINDOW_HEIGHT - 40, centered=True)

    def run(self) -> None:
        running = True
        self.start_new_hand()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE and not self.hand_in_progress:
                        self.start_new_hand()
                    elif event.key == pygame.K_UP:
                        self.decision_delay = max(0.2, self.decision_delay - 0.2)
                    elif event.key == pygame.K_DOWN:
                        self.decision_delay = min(2.0, self.decision_delay + 0.2)
            
            self.update_game_state()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python demo.py <path_to_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    demo = BlackjackDemo(model_path)
    demo.run()