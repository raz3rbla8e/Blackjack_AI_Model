from collections import defaultdict
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any

from agents.base_agent import BaseAgent
from game.hand import Hand
from game.deck import Deck
from game.card import Card

class QLearningAgent(BaseAgent):
    def __init__(self, 
                alpha: float = 0.1,
                gamma: float = 0.9,
                epsilon: float = 0.1,
                epsilon_decay: float = 0.99995,
                min_epsilon: float = 0.01,
                learning_rate_decay: float = 0.99999,
                batch_size: int = 64,
                buffer_size: int = 100000) -> None:
        #q tables
        self.Q1 = defaultdict(float)
        self.Q2 = defaultdict(float)
        #count state visit counts
        self.visit_counts = defaultdict(int)
        #hyperparameters
        self.alpha = alpha
        self.initial_alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        #store initial parameters
        self.initial_params = {
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'min_epsilon': min_epsilon,
            'learning_rate_decay': learning_rate_decay,
            'batch_size': batch_size,
            'buffer_size': buffer_size
        }

    #Returns overall value of a hand
    def getHandValue(self, cards: List[Card]) -> int:
        total = 0
        aces = 0
        for card in cards:
            if card.getValue() == 1:
                aces += 1
            else:
                total += card.getValue()
        
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
            else:
                total += 1
                
        return total
    
    #Returns state representation
    def getState(self, playerHand: Hand, dealerHand: Hand, gameDeck: Deck) -> Tuple:
        player_cards = playerHand.getHand()
        dealer_cards = dealerHand.getHand()
        
        player_value = self.getHandValue(player_cards)
        dealer_upcard = dealer_cards[0].getValue()
        
        #card counting
        running_count = gameDeck.getTrueCount()
        count_bucket = np.sign(running_count)  # Returns -1, 0, or 1
        
        #ace tracking
        num_aces = sum(1 for card in player_cards if card.getValue() == 1)
        has_usable_ace = num_aces > 0 and player_value + 10 <= 21
        
        #checks if hand is a pair
        is_pair = (len(player_cards) == 2 and 
                  player_cards[0].getValue() == player_cards[1].getValue())
        pair_value = player_cards[0].getValue() if is_pair else 0
        
        return (player_value, has_usable_ace, dealer_upcard, 
                count_bucket, is_pair, pair_value)
    
    #returns the avaliable actions given a state
    def _get_available_actions(self, playerHand: Hand, split_depth: int = 0) -> List[int]:
        if playerHand is None:
            return [0, 1]
            
        available_actions = [0, 1]
        
        #add split if possible
        if (playerHand.canBeSplit() and split_depth < 3 and 
            len(playerHand.getHand()) == 2):
            available_actions.append(2)
            
        return available_actions
    
    #gets the best action from the q table
    def getAction(self, playerHand: Hand, dealerHand: Hand, gameDeck: Deck, split_depth: int = 0) -> int:
        state = self.getState(playerHand, dealerHand, gameDeck)
        available_actions = self._get_available_actions(playerHand, split_depth)
        
        # Exploration
        #FIXME: Why is this here? should this have been moved in the refactor to train.py?
        if np.random.random() < self.epsilon:
            visit_counts = [self.visit_counts[(state, a)] for a in available_actions]
            total_visits = sum(visit_counts) + len(available_actions)
            probs = [(total_visits - count) / total_visits for count in visit_counts]
            probs = [p / sum(probs) for p in probs]
            return np.random.choice(available_actions, p=probs)
        
        #Exploitation: use average of both Q-tables
        q_values = {action: (self.Q1[(state, action)] + self.Q2[(state, action)]) / 2 
                   for action in available_actions}
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def _get_dynamic_alpha(self, state: Tuple, action: int) -> float:
        visit_count = self.visit_counts[(state, action)]
        decay = self.learning_rate_decay ** visit_count
        return max(self.alpha * decay, 0.01)
    #updates an entry in the q table
    def update_Q(self, state: Tuple, action: int, reward: float, next_state: Tuple, terminal: bool) -> float:
        self.visit_counts[(state, action)] += 1
        dynamic_alpha = self._get_dynamic_alpha(state, action)
        
        #r
        if np.random.random() < 0.5:
            target = self._compute_target(reward, next_state, terminal, self.Q1, self.Q2)
            current_q = self.Q1[(state, action)]
            self.Q1[(state, action)] = current_q + dynamic_alpha * (target - current_q)
            return target - current_q
        else:
            target = self._compute_target(reward, next_state, terminal, self.Q2, self.Q1)
            current_q = self.Q2[(state, action)]
            self.Q2[(state, action)] = current_q + dynamic_alpha * (target - current_q)
            return target - current_q
    
    def _compute_target(self, reward: float, next_state: Tuple, terminal: bool,
                       q_target: Dict, q_select: Dict) -> float:
        if terminal:
            return reward
        
        # Use one Q-table to select action that maximizes the other Q-table
        next_actions = [0, 1]  # Basic actions always available
        next_q_values = {a: q_select[(next_state, a)] for a in next_actions}
        best_action = max(next_q_values.items(), key=lambda x: x[1])[0]
        return reward + self.gamma * q_target[(next_state, best_action)]
    
    #Returns visualization of current agent strategy
    def get_strategy_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hard_totals = np.zeros((10, 10))  # Player 12-21 vs Dealer 2-11
        soft_totals = np.zeros((10, 10))  # Soft hands
        pairs = np.zeros((10, 10))        # Pairs
        
        # Fill matrices with current strategy
        for player_value in range(12, 22):
            for dealer_up in range(2, 12):
                # Hard totals
                state = (player_value, False, dealer_up, 0, False, 0)
                q_values = {action: (self.Q1[(state, action)] + self.Q2[(state, action)]) / 2 
                          for action in [0, 1]}
                action = max(q_values.items(), key=lambda x: x[1])[0]
                hard_totals[player_value-12][dealer_up-2] = action
                
                # Soft totals
                state = (player_value, True, dealer_up, 0, False, 0)
                q_values = {action: (self.Q1[(state, action)] + self.Q2[(state, action)]) / 2 
                          for action in [0, 1]}
                action = max(q_values.items(), key=lambda x: x[1])[0]
                soft_totals[player_value-12][dealer_up-2] = action
        
        # Pairs
        for card_value in range(2, 12):
            for dealer_up in range(2, 12):
                state = (card_value * 2, False, dealer_up, 0, True, card_value)
                q_values = {action: (self.Q1[(state, action)] + self.Q2[(state, action)]) / 2 
                          for action in [0, 1, 2]}
                action = max(q_values.items(), key=lambda x: x[1])[0]
                pairs[card_value-2][dealer_up-2] = action
        
        return hard_totals, soft_totals, pairs
    #saves the current agent to a json file
    def save(self, filepath: str) -> None:
        q1_table_serializable = {str(state_action): value 
                            for state_action, value in self.Q1.items()}
        q2_table_serializable = {str(state_action): value 
                            for state_action, value in self.Q2.items()}
        
        save_data = {
            'q1_table': q1_table_serializable,
            'q2_table': q2_table_serializable,
            'parameters': self.initial_params,
            'training_info': {
                'current_epsilon': self.epsilon,
                'current_alpha': self.alpha,
                'visit_counts': {str(k): v for k, v in self.visit_counts.items()},
                'batch_size': self.batch_size,
                'buffer_size': self.buffer_size
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

    #Loads pretrained agent from json file
    @classmethod
    def load(cls, filepath: str) -> 'QLearningAgent':
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        agent = cls(**save_data['parameters'])
        
        #load Q-tables
        for state_action_str, value in save_data['q1_table'].items():
            state_action = eval(state_action_str)
            agent.Q1[state_action] = value
        for state_action_str, value in save_data['q2_table'].items():
            state_action = eval(state_action_str)
            agent.Q2[state_action] = value
        
        #load training state
        agent.epsilon = save_data['training_info']['current_epsilon']
        agent.alpha = save_data['training_info']['current_alpha']
        # agent.batch_size = save_data['training_info']['batch_size']
        agent.batch_size = 64
        # agent.buffer_size = save_data['training_info']['buffer_size']
        agent.buffer_size = 500000
        
        #load visit counts
        agent.visit_counts = defaultdict(int)
        for state_action_str, count in save_data['training_info']['visit_counts'].items():
            agent.visit_counts[eval(state_action_str)] = count
        
        return agent

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)