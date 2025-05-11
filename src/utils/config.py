from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class TrainingConfig:
    mode: int = 1  # 0: quick test, 1: best known parameters, 2: full grid search
    base_episodes: int = 500000
    batch_size: int = 64
    buffer_size: int = 100000
    save_intervals: int = 10000
    
    # Evaluation settings
    eval_games: int = 1000
    eval_intervals: int = 5000
    
    # File paths
    results_dir: str = 'training_results'
    model_dir: str = 'saved_models'
    
    def __post_init__(self):
        #make directories needed for training
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
    @property
    def param_grid(self) -> Dict[str, list]:
        if self.mode == 0:  #e2e test mode
            return {
                'alpha': [0.1, 0.2],
                'gamma': [0.95, 0.99],
                'epsilon': [0.1],
                'epsilon_decay': [0.9999],
                'min_epsilon': [0.01],
                'learning_rate_decay': [0.9999],
                'batch_size': [32],
                'buffer_size': [10000]
            }
        elif self.mode == 1:  #best known parameters
            return {
                'alpha': [0.001],
                'gamma': [0.97],
                'epsilon': [0.05],
                'epsilon_decay': [0.99995],
                'min_epsilon': [0.01],
                'learning_rate_decay': [0.9999],
                'batch_size': [64],
                'buffer_size': [50000]
            }
        #TODO: Shrink hyperparameter space
        else:  #full grid search
            return {
                'alpha': [0.001, 0.01, 0.1, 0.2],
                'gamma': [0.95, 0.97, 0.99],
                'epsilon': [0.05, 0.1, 0.2],
                'epsilon_decay': [0.9999, 0.99995, 0.99999],
                'min_epsilon': [0.01],
                'learning_rate_decay': [0.9999, 0.99995, 0.99999],
                'batch_size': [64],
                'buffer_size': [50000,100000]
            }

@dataclass
class GameConfig:
    max_splits: int = 3
    default_bet: float = 1.0
    blackjack_payout: float = 1.5
    minimum_dealer_stand: int = 17
    reshuffle_threshold: int = 15