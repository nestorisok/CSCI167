"""
Analysis Code for Fashion-MNIST Experiments
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

# Load data
df = pd.read_csv('logs_deep.csv')

# Parse epoch lists
df['epoch_test_accuracies'] = df['epoch_test_accuracies'].apply(ast.literal_eval)
df['epoch_train_losses'] = df['epoch_train_losses'].apply(ast.literal_eval)
df['epoch_test_losses'] = df['epoch_test_losses'].apply(ast.literal_eval)

# =============================================================================
# 1. OPTIMIZER COMPARISON
# =============================================================================
print("="*60)
print("OPTIMIZER COMPARISON")
print("="*60)

sgd_best = df[df['optimizer'] == 'SGD']['final_test_accuracy'].max()
adam_best = df[df['optimizer'] == 'Adam']['final_test_accuracy'].max()

print(f"SGD Best Accuracy: {sgd_best*100:.2f}%")
print(f"Adam Best Accuracy: {adam_best*100:.2f}%")
print(f"Difference: {(adam_best - sgd_best)*100:.2f}%\n")

# Average by optimizer
print("Average Accuracy by Optimizer:")
print(df.groupby('optimizer')['final_test_accuracy'].mean() * 100)


# =============================================================================
# 2. NORMALIZATION COMPARISON
# =============================================================================
print("\n" + "="*60)
print("NORMALIZATION COMPARISON")
print("="*60)

norm_comparison = df.groupby(['optimizer', 'normalization'])['final_test_accuracy'].agg(['mean', 'max'])
print(norm_comparison * 100)


# =============================================================================
# 3. LEARNING RATE EFFECTS
# =============================================================================
print("\n" + "="*60)
print("LEARNING RATE EFFECTS")
print("="*60)

lr_comparison = df.groupby(['optimizer', 'lr'])['final_test_accuracy'].agg(['mean', 'max'])
print(lr_comparison * 100)


# =============================================================================
# 4. VISUALIZATIONS
# =============================================================================

# Plot 1: Learning Curves for Best Configurations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Best SGD
best_sgd_idx = df[df['optimizer'] == 'SGD']['final_test_accuracy'].idxmax()
best_sgd = df.loc[best_sgd_idx]

axes[0,0].plot([acc*100 for acc in best_sgd['epoch_test_accuracies']], 
               marker='o', linewidth=2, label='SGD Best')
axes[0,0].set_title(f"Best SGD: {best_sgd['final_test_accuracy']*100:.2f}%\n"
                    f"(lr={best_sgd['lr']}, norm={best_sgd['normalization']})")
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Test Accuracy (%)')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_ylim([75, 92])

# Best Adam
best_adam_idx = df[df['optimizer'] == 'Adam']['final_test_accuracy'].idxmax()
best_adam = df.loc[best_adam_idx]

axes[0,1].plot([acc*100 for acc in best_adam['epoch_test_accuracies']], 
               marker='s', linewidth=2, label='Adam Best', color='orange')
axes[0,1].set_title(f"Best Adam: {best_adam['final_test_accuracy']*100:.2f}%\n"
                    f"(lr={best_adam['lr']}, norm={best_adam['normalization']})")
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Test Accuracy (%)')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_ylim([75, 92])

# Plot 2: Normalization Effect
norm_data = df.groupby(['optimizer', 'normalization'])['final_test_accuracy'].mean().unstack() * 100
norm_data.plot(kind='bar', ax=axes[1,0], rot=0)
axes[1,0].set_title('Normalization Impact on Accuracy')
axes[1,0].set_ylabel('Test Accuracy (%)')
axes[1,0].set_xlabel('Optimizer')
axes[1,0].legend(title='Normalization')
axes[1,0].grid(True, alpha=0.3, axis='y')

# Plot 3: Learning Rate Effect
lr_data = df.groupby(['optimizer', 'lr'])['final_test_accuracy'].mean().unstack() * 100
lr_data.plot(kind='bar', ax=axes[1,1], rot=0)
axes[1,1].set_title('Learning Rate Impact on Accuracy')
axes[1,1].set_ylabel('Test Accuracy (%)')
axes[1,1].set_xlabel('Optimizer')
axes[1,1].legend(title='Learning Rate')
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hyperparameter_analysis_deep.png', dpi=300, bbox_inches='tight')
plt.show()


# =============================================================================
# 5. BATCH SIZE ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("BATCH SIZE COMPARISON")
print("="*60)

batch_comparison = df.groupby(['optimizer', 'batch_size'])['final_test_accuracy'].agg(['mean', 'max'])
print(batch_comparison * 100)


# =============================================================================
# 6. REGULARIZATION ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("REGULARIZATION EFFECTS")
print("="*60)

reg_comparison = df.groupby(['optimizer', 'dropout', 'weight_decay'])['final_test_accuracy'].mean()
print(reg_comparison.unstack() * 100)


# =============================================================================
# 7. TOP 10 CONFIGURATIONS
# =============================================================================
print("\n" + "="*60)
print("TOP 10 CONFIGURATIONS")
print("="*60)

top_10 = df.nlargest(10, 'final_test_accuracy')[
    ['optimizer', 'lr', 'normalization', 'batch_size', 'dropout', 
     'weight_decay', 'final_test_accuracy']
].copy()
top_10['final_test_accuracy'] = top_10['final_test_accuracy'] * 100
print(top_10.to_string(index=False))


# =============================================================================
# 8. CONVERGENCE SPEED ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("CONVERGENCE SPEED: Epochs to Reach 85% Accuracy")
print("="*60)

def epochs_to_target(acc_list, target=0.85):
    """Find first epoch that reaches target accuracy"""
    for i, acc in enumerate(acc_list):
        if acc >= target:
            return i + 1
    return len(acc_list)  # Never reached

df['epochs_to_85'] = df['epoch_test_accuracies'].apply(epochs_to_target)

convergence_comparison = df.groupby(['optimizer', 'lr'])['epochs_to_85'].agg(['mean', 'min'])
print(convergence_comparison)


# =============================================================================
# 9. STATISTICAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)

print(f"Total experiments: {len(df)}")
print(f"Best accuracy: {df['final_test_accuracy'].max()*100:.2f}%")
print(f"Worst accuracy: {df['final_test_accuracy'].min()*100:.2f}%")
print(f"Mean accuracy: {df['final_test_accuracy'].mean()*100:.2f}%")
print(f"Std accuracy: {df['final_test_accuracy'].std()*100:.2f}%")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)