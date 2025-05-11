from typing import List, Tuple, Dict, Any
import numpy as np
from itertools import product
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import json
from datetime import datetime
import pandas as pd

from agents.q_learning_agent import QLearningAgent
from training.train import BlackjackTrainer
from agents.evaluation import evaluate_agent
from training.visualization import plot_learning_curves, plot_parameter_importance
from training.analysis import analyze_agent_strategy
from utils.config import TrainingConfig

class HyperparameterOptimizer:
    def __init__(self, config: TrainingConfig):
        self.param_grid = config.param_grid
        self.base_episodes = config.base_episodes
        self.eval_games = config.eval_games
        self.eval_intervals = config.eval_intervals
        self.config = config
        
        # Results storage
        self.results_dir = config.results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def _run_single_config(self, args: Tuple[Dict, str]) -> tuple:
        config, _ = args
        #pass params to agent
        agent = QLearningAgent(**config)
        #create trainer for agent
        trainer = BlackjackTrainer(agent)
        
        history = {
            'win_rate': [],
            'avg_reward': [],
            'split_rate': [],
            'episodes': [],
            'bust_rate': [],
            'blackjack_rate': []
        }
        
        eval_step = self.base_episodes // self.eval_intervals
        total_episodes = 0
        
        for _ in range(self.eval_intervals):
            metrics_list = trainer.train(numEpisodes=eval_step)
            total_episodes += eval_step
            
            # Calculate average metrics for this interval
            avg_metrics = {
                'wins': sum(m.wins for m in metrics_list) / len(metrics_list),
                'splits': sum(m.splits for m in metrics_list) / len(metrics_list),
                'blackjacks': sum(m.blackjacks for m in metrics_list) / len(metrics_list),
                'episode_reward': sum(m.episode_reward for m in metrics_list) / len(metrics_list)
            }
            
            # Evaluate and save metrics
            eval_metrics = evaluate_agent(agent, self.eval_games)
            history['episodes'].append(total_episodes)
            history['win_rate'].append(eval_metrics['win_rate'])
            history['avg_reward'].append(eval_metrics['avg_reward'])
            history['split_rate'].append(eval_metrics['split_rate'])
            history['bust_rate'].append(eval_metrics['bust_rate'])
            history['blackjack_rate'].append(eval_metrics['blackjack_rate'])
        
        # Final evaluation with more games
        final_metrics = evaluate_agent(agent, self.eval_games * 2)
        
        return config, history, final_metrics, agent

    def analyze_results(self, all_results: List[tuple]) -> None:
        # Create pandas DataFrame from results
        results_list = []
        for config, _, metrics, _ in all_results:
            result_dict = config.copy()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    result_dict['final_' + k] = v
            results_list.append(result_dict)
        
        results_df = pd.DataFrame(results_list)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'{self.results_dir}/hyperparameter_results_{timestamp}.csv')
        
        # Visualize
        plot_learning_curves(all_results, self.results_dir)
        plot_parameter_importance(results_df, self.param_grid, self.results_dir)
        
        # Get best model
        best_result = max(all_results, key=lambda x: x[2]['win_rate'])
        best_config, _, best_metrics, best_agent = best_result
        
        # Save best model
        model_path = f'{self.results_dir}/best_model_{timestamp}.json'
        best_agent.save(model_path)
        
        # Save best hyperparameters
        summary = {
            'model_path': model_path,
            'configuration': best_config,
            'metrics': {k: v for k, v in best_metrics.items() 
                       if isinstance(v, (int, float))}
        }
        
        with open(f'{self.results_dir}/best_model_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Visualize best strategy
        analyze_agent_strategy(best_agent, self.results_dir)
        
        print(f"\nBest model saved to: {model_path}")

    def run_optimization(self) -> None:
        param_combinations = list(ParameterGrid(self.param_grid))
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Create timestamp for this optimization run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare arguments for parallel processing
        args = [(config, None) for config in param_combinations]
        
        # Check # of threads available and parallelize
        num_processes = min(cpu_count() - 1, len(param_combinations))
        print(f"Using {num_processes} processes for parallel training...")
        
        # Multithread
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._run_single_config, args),
                total=len(param_combinations),
                desc="Training configurations"
            ))
            
        self.analyze_results(results)
        
        best_config = max(results, key=lambda x: x[2]['win_rate'])
        print("\nBest Configuration Found:")
        print(json.dumps(best_config[0], indent=2))
        print("\nBest Configuration Metrics:")
        print(json.dumps({k: v for k, v in best_config[2].items() 
                         if isinstance(v, (int, float))}, indent=2))