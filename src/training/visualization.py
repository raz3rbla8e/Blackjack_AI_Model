import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict

def plot_learning_curves(all_results: List[tuple], results_dir: str) -> None:
    """Plot learning curves with fixed data handling"""
    metrics = ['win_rate', 'avg_reward', 'split_rate', 
              'bust_rate', 'blackjack_rate']
    
    #sort results by final win rate to find best models
    sorted_results = sorted(all_results, 
                          key=lambda x: x[2]['win_rate'], 
                          reverse=True)
    
    #grab the top 10 for plotting
    top_results = sorted_results[:10]
    
    #plot
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        for config, history, _, _ in top_results:
            #handle a weird edge case
            if metric not in history or 'episodes' not in history:
                continue
                
            episodes = history['episodes']
            values = history[metric]
            
            #i'm not sure why this happens sometimes, it seems stochastic
            if len(episodes) != len(values):
                print(f"Warning: Data misalignment for {metric}. Episodes: {len(episodes)}, Values: {len(values)}")
                min_len = min(len(episodes), len(values))
                episodes = episodes[:min_len]
                values = values[:min_len]
            
            if len(episodes) == 0:
                continue
            
            #sample the points for computational efficiency
            if len(episodes) > 100:
                indices = np.linspace(0, len(episodes)-1, 100, dtype=int)
                episodes = [episodes[i] for i in indices]
                values = [values[i] for i in indices]
            
            label = f"α={config['alpha']:.3f}, γ={config['gamma']:.3f}"
            plt.plot(episodes, values, label=label, alpha=0.7)
        
        plt.xlabel('Episodes')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Top 10 Configurations - {metric.replace("_", " ").title()}')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        #save the plots
        plt.tight_layout()
        plt.savefig(f'{results_dir}/learning_curve_{metric}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4*len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        #handle another weird edge case
        for config, history, _, _ in sorted_results[:3]:
            if metric not in history or 'episodes' not in history:
                continue
                
            episodes = history['episodes']
            values = history[metric]
            
            # make sure data is aligned
            if len(episodes) != len(values):
                min_len = min(len(episodes), len(values))
                episodes = episodes[:min_len]
                values = values[:min_len]
            
            if len(episodes) == 0:
                continue
            
            #sample again
            if len(episodes) > 100:
                indices = np.linspace(0, len(episodes)-1, 100, dtype=int)
                episodes = [episodes[i] for i in indices]
                values = [values[i] for i in indices]
            
            label = f"α={config['alpha']:.3f}, γ={config['gamma']:.3f}"
            ax.plot(episodes, values, label=label, alpha=0.7)
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/learning_curves_summary.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

#determine which metrics are most important for model success
def plot_parameter_importance(results_df: pd.DataFrame, param_grid: Dict, results_dir: str) -> None:
    metrics = ['final_win_rate', 'final_avg_reward', 
               'final_blackjack_rate', 'final_bust_rate']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        if metric not in results_df.columns:
            print(f"Warning: Metric {metric} not found in results")
            continue
            
        # Create parameter importance plot
        param_impacts = {}
        for param in param_grid.keys():
            if param not in results_df.columns:
                print(f"Warning: Parameter {param} not found in results")
                continue
                
            values = results_df[param].unique()
            means = [results_df[results_df[param] == val][metric].mean() 
                    for val in values]
            param_impacts[param] = np.std(means)
        
        if not param_impacts:
            continue
            
        #sort parameters by impact
        sorted_impacts = sorted(param_impacts.items(), 
                              key=lambda x: x[1], reverse=True)
        #mnake the grapg
        ax = axes[i]
        bars = sns.barplot(x=[x[0] for x in sorted_impacts],
                          y=[x[1] for x in sorted_impacts],
                          ax=ax)
        ax.set_xticks(range(len(sorted_impacts)))
        ax.set_xticklabels([x[0] for x in sorted_impacts], rotation=45, ha='right')
        ax.set_title(f'Parameter Impact on {metric.replace("_", " ").title()}')
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/parameter_importance.png', 
               bbox_inches='tight', dpi=300)
    plt.close()