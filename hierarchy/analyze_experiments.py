"""
Experiment Analysis and Comparison Tool

This script helps analyze and compare different experiments to identify
the best hyperparameter configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

def load_experiments_summary():
    """Load the experiments summary CSV file."""
    summary_path = Path('experiments') / 'experiments_summary.csv'
    if not summary_path.exists():
        print("No experiments summary found. Run some experiments first!")
        return None
    
    df = pd.read_csv(summary_path)
    print(f"Loaded {len(df)} experiments")
    return df

def print_experiments_summary(df):
    """Print a summary of all experiments."""
    print("\n" + "=" * 80)
    print("EXPERIMENTS SUMMARY")
    print("=" * 80)
    
    # Sort by best validation accuracy
    df_sorted = df.sort_values('best_val_accuracy', ascending=False)
    
    for _, row in df_sorted.iterrows():
        status_icon = "✅" if row['status'] == 'completed' else "🔄" if row['status'] == 'running' else "❌"
        print(f"{status_icon} {row['experiment_name']:<20} | "
              f"LR: {row['learning_rate']:<8} | "
              f"Batch: {row['batch_size']:<4} | "
              f"Cls Weight: {row['classification_weight']:<6} | "
              f"Best Acc: {row['best_val_accuracy']:.4f} | "
              f"Best Loss: {row['best_val_loss']:.4f}")

def plot_hyperparameter_comparison(df):
    """Create plots comparing different hyperparameters."""
    if df is None or len(df) < 2:
        print("Need at least 2 experiments to create comparison plots")
        return
    
    # Filter completed experiments
    completed_df = df[df['status'] == 'completed'].copy()
    if len(completed_df) < 2:
        print("Need at least 2 completed experiments to create comparison plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Hyperparameter Comparison Across Experiments', fontsize=16)
    
    # Learning rate vs accuracy
    axes[0, 0].scatter(completed_df['learning_rate'], completed_df['best_val_accuracy'], 
                       s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Best Validation Accuracy')
    axes[0, 0].set_title('Learning Rate vs Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Batch size vs accuracy
    axes[0, 1].scatter(completed_df['batch_size'], completed_df['best_val_accuracy'], 
                       s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Best Validation Accuracy')
    axes[0, 1].set_title('Batch Size vs Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Classification weight vs accuracy
    axes[1, 0].scatter(completed_df['classification_weight'], completed_df['best_val_accuracy'], 
                       s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Classification Weight')
    axes[1, 0].set_ylabel('Best Validation Accuracy')
    axes[1, 0].set_title('Classification Weight vs Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss vs accuracy
    axes[1, 1].scatter(completed_df['best_val_loss'], completed_df['best_val_accuracy'], 
                       s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Best Validation Loss')
    axes[1, 1].set_ylabel('Best Validation Accuracy')
    axes[1, 1].set_title('Loss vs Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path('experiments') / 'hyperparameter_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Hyperparameter comparison plot saved to: {plot_path}")
    
    plt.show()

def find_best_experiment(df):
    """Find the best performing experiment."""
    if df is None:
        return None
    
    completed_df = df[df['status'] == 'completed'].copy()
    if len(completed_df) == 0:
        print("No completed experiments found")
        return None
    
    best_idx = completed_df['best_val_accuracy'].idxmax()
    best_exp = completed_df.loc[best_idx]
    
    print("\n" + "=" * 60)
    print("🏆 BEST EXPERIMENT")
    print("=" * 60)
    print(f"Name: {best_exp['experiment_name']}")
    print(f"Best Validation Accuracy: {best_exp['best_val_accuracy']:.4f}")
    print(f"Best Validation Loss: {best_exp['best_val_loss']:.4f}")
    print(f"Learning Rate: {best_exp['learning_rate']}")
    print(f"Batch Size: {best_exp['batch_size']}")
    print(f"Classification Weight: {best_exp['classification_weight']}")
    print(f"Embedding Loss Type: {best_exp['embedding_loss_type']}")
    print(f"Model Type: {best_exp['model_type']}")
    print(f"Embedding Dimension: {best_exp['embedding_dim']}")
    
    return best_exp

def analyze_experiment_details(experiment_name):
    """Analyze the details of a specific experiment."""
    exp_dir = Path('experiments') / experiment_name
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    print(f"\n📊 DETAILED ANALYSIS: {experiment_name}")
    print("=" * 50)
    
    # Load config
    config_path = exp_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Load metrics
    metrics_path = exp_dir / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nTraining Metrics:")
        print(f"  Total epochs: {len(metrics['train_losses'])}")
        print(f"  Final train accuracy: {metrics['train_accuracies'][-1]:.4f}")
        print(f"  Final val accuracy: {metrics['val_accuracies'][-1]:.4f}")
        print(f"  Final train loss: {metrics['train_losses'][-1]:.4f}")
        print(f"  Final val loss: {metrics['val_losses'][-1]:.4f}")
        
        # Plot training curves
        plot_training_curves(experiment_name, metrics)
    
    # Check for model files
    model_dir = exp_dir / 'model_checkpoints'
    if model_dir.exists():
        model_files = list(model_dir.glob('*.pth'))
        print(f"\nModel Checkpoints:")
        for model_file in model_files:
            print(f"  {model_file.name}")

def plot_training_curves(experiment_name, metrics):
    """Plot training curves for a specific experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16)
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, metrics['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, metrics['val_losses'], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, metrics['train_accuracies'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(epochs, metrics['val_accuracies'], label='Val Accuracy', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Component losses
    if 'embedding_losses' in metrics and 'classification_losses' in metrics:
        axes[1, 0].plot(epochs, metrics['embedding_losses'], label='Embedding Loss', color='green')
        axes[1, 0].plot(epochs, metrics['classification_losses'], label='Classification Loss', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Component Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rates' in metrics:
        axes[1, 1].plot(epochs, metrics['learning_rates'], label='Learning Rate', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    exp_dir = Path('experiments') / experiment_name
    plot_path = exp_dir / 'plots' / 'detailed_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Detailed analysis plot saved to: {plot_path}")
    
    plt.show()

def main():
    """Main analysis function."""
    print("🔬 EXPERIMENT ANALYSIS TOOL")
    print("=" * 40)
    
    # Load experiments summary
    df = load_experiments_summary()
    if df is None:
        return
    
    while True:
        print("\nOptions:")
        print("1. View experiments summary")
        print("2. Plot hyperparameter comparison")
        print("3. Find best experiment")
        print("4. Analyze specific experiment")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print_experiments_summary(df)
        
        elif choice == '2':
            plot_hyperparameter_comparison(df)
        
        elif choice == '3':
            find_best_experiment(df)
        
        elif choice == '4':
            experiment_name = input("Enter experiment name: ").strip()
            analyze_experiment_details(experiment_name)
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 