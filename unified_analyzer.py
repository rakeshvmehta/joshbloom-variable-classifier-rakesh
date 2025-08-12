"""
Unified Experiment Analysis Tool

This tool can analyze and compare experiments from both hierarchical and non-hierarchical approaches,
providing a comprehensive view of all experiments across the project.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import numpy as np

class UnifiedAnalyzer:
    """Unified analyzer for both hierarchical and non-hierarchical experiments."""
    
    def __init__(self):
        self.hierarchy_summary = None
        self.non_hierarchy_summary = None
        self.all_experiments = None
        
    def load_all_experiments(self):
        """Load experiments from both approaches."""
        print("Loading experiments from both approaches...")
        
        # Load hierarchical experiments
        hierarchy_path = Path('hierarchy/experiments/experiments_summary.csv')
        if hierarchy_path.exists():
            self.hierarchy_summary = pd.read_csv(hierarchy_path)
            print(f"Loaded {len(self.hierarchy_summary)} hierarchical experiments")
        else:
            self.hierarchy_summary = pd.DataFrame()
            print("No hierarchical experiments found")
        
        # Load non-hierarchical experiments
        non_hierarchy_path = Path('non_hierarchical/experiments/experiments_summary.csv')
        if non_hierarchy_path.exists():
            self.non_hierarchy_summary = pd.read_csv(non_hierarchy_path)
            print(f"Loaded {len(self.non_hierarchy_summary)} non-hierarchical experiments")
        else:
            self.non_hierarchy_summary = pd.DataFrame()
            print("No non-hierarchical experiments found")
        
        # Combine all experiments
        if not self.hierarchy_summary.empty and not self.non_hierarchy_summary.empty:
            self.all_experiments = pd.concat([self.hierarchy_summary, self.non_hierarchy_summary], ignore_index=True)
        elif not self.hierarchy_summary.empty:
            self.all_experiments = self.hierarchy_summary.copy()
        elif not self.non_hierarchy_summary.empty:
            self.all_experiments = self.non_hierarchy_summary.copy()
        else:
            self.all_experiments = pd.DataFrame()
        
        if not self.all_experiments.empty:
            print(f"Total experiments: {len(self.all_experiments)}")
            print(f"Approaches: {self.all_experiments['approach_type'].value_counts().to_dict()}")
    
    def print_unified_summary(self):
        """Print a unified summary of all experiments."""
        if self.all_experiments is None or self.all_experiments.empty:
            print("No experiments found. Run some experiments first!")
            return
        
        print("\n" + "=" * 100)
        print("UNIFIED EXPERIMENTS SUMMARY")
        print("=" * 100)
        
        # Sort by best validation accuracy
        df_sorted = self.all_experiments.sort_values('best_val_accuracy', ascending=False)
        
        for _, row in df_sorted.iterrows():
            status_icon = "✅" if row['status'] == 'completed' else "🔄" if row['status'] == 'running' else "❌"
            approach_icon = "🎯" if row['approach_type'] == 'hierarchical' else "📊"
            
            print(f"{status_icon} {approach_icon} {row['experiment_name']:<25} | "
                  f"Type: {row['approach_type']:<15} | "
                  f"Model: {row['model_type']:<10} | "
                  f"LR: {row['learning_rate']:<8} | "
                  f"Batch: {row['batch_size']:<4} | "
                  f"Best Acc: {row['best_val_accuracy']:.4f} | "
                  f"Best Loss: {row['best_val_loss']:.4f}")
    
    def plot_approach_comparison(self):
        """Plot comparison between hierarchical and non-hierarchical approaches."""
        if self.all_experiments is None or self.all_experiments.empty:
            print("No experiments found for comparison")
            return
        
        # Filter completed experiments
        completed_df = self.all_experiments[self.all_experiments['status'] == 'completed'].copy()
        if len(completed_df) < 2:
            print("Need at least 2 completed experiments for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Unified Experiment Analysis - All Approaches', fontsize=16)
        
        # Approach performance comparison
        approach_stats = completed_df.groupby('approach_type').agg({
            'best_val_accuracy': ['mean', 'std', 'count'],
            'best_val_loss': ['mean', 'std']
        }).round(4)
        
        # Accuracy comparison
        axes[0, 0].bar(completed_df['approach_type'].unique(), 
                       completed_df.groupby('approach_type')['best_val_accuracy'].mean(),
                       yerr=completed_df.groupby('approach_type')['best_val_accuracy'].std(),
                       capsize=5, alpha=0.7)
        axes[0, 0].set_title('Average Performance by Approach')
        axes[0, 0].set_ylabel('Best Validation Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate vs accuracy by approach
        for approach in completed_df['approach_type'].unique():
            approach_data = completed_df[completed_df['approach_type'] == approach]
            axes[0, 1].scatter(approach_data['learning_rate'], approach_data['best_val_accuracy'], 
                              label=approach, alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Learning Rate')
        axes[0, 1].set_ylabel('Best Validation Accuracy')
        axes[0, 1].set_title('Learning Rate vs Accuracy by Approach')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Batch size vs accuracy by approach
        for approach in completed_df['approach_type'].unique():
            approach_data = completed_df[completed_df['approach_type'] == approach]
            axes[1, 0].scatter(approach_data['batch_size'], approach_data['best_val_accuracy'], 
                              label=approach, alpha=0.7, s=100)
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('Best Validation Accuracy')
        axes[1, 0].set_title('Batch Size vs Accuracy by Approach')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss vs accuracy by approach
        for approach in completed_df['approach_type'].unique():
            approach_data = completed_df[completed_df['approach_type'] == approach]
            axes[1, 1].scatter(approach_data['best_val_loss'], approach_data['best_val_accuracy'], 
                              label=approach, alpha=0.7, s=100)
        axes[1, 1].set_xlabel('Best Validation Loss')
        axes[1, 1].set_ylabel('Best Validation Accuracy')
        axes[1, 1].set_title('Loss vs Accuracy by Approach')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path('unified_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Unified analysis plot saved to: {plot_path}")
        
        plt.show()
    
    def find_best_experiments(self):
        """Find the best experiments across all approaches."""
        if self.all_experiments is None or self.all_experiments.empty:
            print("No experiments found")
            return
        
        completed_df = self.all_experiments[self.all_experiments['status'] == 'completed'].copy()
        if completed_df.empty:
            print("No completed experiments found")
            return
        
        print("\n" + "=" * 80)
        print("🏆 BEST EXPERIMENTS ACROSS ALL APPROACHES")
        print("=" * 80)
        
        # Overall best
        overall_best = completed_df.loc[completed_df['best_val_accuracy'].idxmax()]
        print(f"🥇 OVERALL BEST: {overall_best['experiment_name']}")
        print(f"   Approach: {overall_best['approach_type']}")
        print(f"   Model: {overall_best['model_type']}")
        print(f"   Best Accuracy: {overall_best['best_val_accuracy']:.4f}")
        print(f"   Best Loss: {overall_best['best_val_loss']:.4f}")
        print(f"   Learning Rate: {overall_best['learning_rate']}")
        print(f"   Batch Size: {overall_best['batch_size']}")
        
        # Best by approach
        print(f"\n🏆 BEST BY APPROACH:")
        for approach in completed_df['approach_type'].unique():
            approach_data = completed_df[completed_df['approach_type'] == approach]
            best_in_approach = approach_data.loc[approach_data['best_val_accuracy'].idxmax()]
            print(f"\n   {approach.upper()}: {best_in_approach['experiment_name']}")
            print(f"      Accuracy: {best_in_approach['best_val_accuracy']:.4f}")
            print(f"      Loss: {best_in_approach['best_val_loss']:.4f}")
            print(f"      Model: {best_in_approach['model_type']}")
    
    def analyze_specific_experiment(self, experiment_name):
        """Analyze a specific experiment in detail."""
        # Search in both approaches
        experiment_found = False
        
        for approach in ['hierarchical', 'non_hierarchical']:
            exp_dir = Path(f'{approach}/experiments/{experiment_name}')
            if exp_dir.exists():
                experiment_found = True
                print(f"\n📊 DETAILED ANALYSIS: {experiment_name} ({approach})")
                print("=" * 60)
                
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
                    self._plot_experiment_curves(experiment_name, metrics, approach)
                
                # Check for model files
                model_dir = exp_dir / 'model_checkpoints'
                if model_dir.exists():
                    model_files = list(model_dir.glob('*.pth'))
                    print(f"\nModel Checkpoints:")
                    for model_file in model_files:
                        print(f"  {model_file.name}")
                
                break
        
        if not experiment_found:
            print(f"Experiment '{experiment_name}' not found in either approach")
    
    def _plot_experiment_curves(self, experiment_name, metrics, approach):
        """Plot training curves for a specific experiment."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {experiment_name} ({approach})', fontsize=16)
        
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
        
        # Learning rate
        if 'learning_rates' in metrics:
            axes[1, 0].plot(epochs, metrics['learning_rates'], label='Learning Rate', color='purple')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss vs Accuracy
        axes[1, 1].scatter(metrics['val_losses'], metrics['val_accuracies'], color='red', alpha=0.6)
        axes[1, 1].set_xlabel('Validation Loss')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Loss vs Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        exp_dir = Path(f'{approach}/experiments/{experiment_name}')
        plot_path = exp_dir / 'plots' / 'detailed_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Detailed analysis plot saved to: {plot_path}")
        
        plt.show()
    
    def run_interactive_analysis(self):
        """Run interactive analysis session."""
        print("🔬 UNIFIED EXPERIMENT ANALYSIS TOOL")
        print("=" * 50)
        
        # Load all experiments
        self.load_all_experiments()
        
        while True:
            print("\nOptions:")
            print("1. View unified experiments summary")
            print("2. Plot approach comparison")
            print("3. Find best experiments")
            print("4. Analyze specific experiment")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.print_unified_summary()
            
            elif choice == '2':
                self.plot_approach_comparison()
            
            elif choice == '3':
                self.find_best_experiments()
            
            elif choice == '4':
                experiment_name = input("Enter experiment name: ").strip()
                self.analyze_specific_experiment(experiment_name)
            
            elif choice == '5':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function to run unified analysis."""
    analyzer = UnifiedAnalyzer()
    analyzer.run_interactive_analysis()

if __name__ == "__main__":
    main() 