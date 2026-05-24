import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.methods.mlp import MLP
from src.methods.kmeans import KMeans
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, mse_fn

# Set seed for reproducibility
np.random.seed(100)

def load_and_prepare_data(dataset_path="data/features.npz"):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'], feature_data['xtest'], feature_data['ytrainreg'],
        feature_data['ytestreg'], feature_data['ytrainclassif'], feature_data['ytestclassif']
    )

    # Perform 80/20 train/validation split on the training set
    n_train = int(0.8 * train_features.shape[0])
    val_features = train_features[n_train:]
    val_labels_reg = train_labels_reg[n_train:]
    val_labels_classif = train_labels_classif[n_train:]
    
    train_features_split = train_features[:n_train]
    train_labels_reg_split = train_labels_reg[:n_train]
    train_labels_classif_split = train_labels_classif[:n_train]

    # Normalize features
    means = np.mean(train_features_split, axis=0, keepdims=True)
    stds = np.std(train_features_split, axis=0, keepdims=True)
    stds[stds == 0] = 1.0 # prevent division by zero
    
    train_features_norm = normalize_fn(train_features_split, means, stds)
    val_features_norm = normalize_fn(val_features, means, stds)

    return (train_features_norm, val_features_norm, 
            train_labels_reg_split, val_labels_reg, 
            train_labels_classif_split, val_labels_classif)

def tune_kmeans(train_x, val_x, train_y, val_y):
    print("--- Tuning K-Means ---")
    k_values = list(range(1, 21))
    accuracies = []
    f1_scores = []
    
    best_k = 1
    best_acc = 0.0
    best_f1 = 0.0

    for k in k_values:
        kmeans = KMeans(K=k, max_iters=100)
        kmeans.fit(train_x, train_y)
        preds = kmeans.predict(val_x)
        
        acc = accuracy_fn(preds, val_y)
        f1 = macrof1_fn(preds, val_y)
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"K={k:2d} | Val Accuracy: {acc:6.2f}% | Val Macro F1: {f1:6.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_f1 = f1

    print(f"Best K-Means: K={best_k} with Accuracy={best_acc:.2f}%, F1={best_f1:.4f}\n")

    # Plot Accuracy and Macro F1 on dual axes
    fig, ax1 = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    
    color = '#1f77b4'
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(k_values, accuracies, color=color, marker='o', linewidth=2.5, label='Accuracy (%)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(k_values)
    
    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('Validation Macro F1', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(k_values, f1_scores, color=color, marker='s', linestyle='--', linewidth=2, label='Macro F1')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.title('K-Means Performance vs Number of Clusters (K)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig('kmeans_sweep.png', dpi=300)
    plt.close()
    
    return best_k, best_acc, best_f1

def compare_mlp_classification(train_x, val_x, train_y, val_y):
    print("--- Comparing SGD vs Momentum (Classification) ---")
    epochs_list = [50, 100, 200, 500]
    
    # Convert labels to one-hot for MLP training
    y_train_oh = np.zeros((train_y.shape[0], 3))
    y_train_oh[np.arange(y_train_oh.shape[0]), train_y.astype(int)] = 1
    
    lr = 0.01  # Best learning rate identified
    
    acc_sgd = []
    acc_momentum = []
    
    D = train_x.shape[1]
    C = 3
    
    for epochs in epochs_list:
        # SGD (momentum = 0.0)
        mlp_sgd = MLP(dimensions=(D, 64, C), activations=(ReLU, Sigmoid))
        mlp_sgd.fit(train_x, y_train_oh, loss=CrossEntropy, epochs=epochs, batch_size=32, learning_rate=lr, momentum=0.0)
        preds_sgd = mlp_sgd.predict(val_x)
        acc_sgd.append(accuracy_fn(np.argmax(preds_sgd, axis=1), val_y))
        
        # Momentum (momentum = 0.9)
        mlp_mom = MLP(dimensions=(D, 64, C), activations=(ReLU, Sigmoid))
        mlp_mom.fit(train_x, y_train_oh, loss=CrossEntropy, epochs=epochs, batch_size=32, learning_rate=lr, momentum=0.9)
        preds_mom = mlp_mom.predict(val_x)
        acc_momentum.append(accuracy_fn(np.argmax(preds_mom, axis=1), val_y))
        
        print(f"Epochs: {epochs:<3d} | SGD Acc: {acc_sgd[-1]:.2f}% | Momentum Acc: {acc_momentum[-1]:.2f}%")
        
    # Generate line plot
    plt.figure(figsize=(7, 4.5))
    sns.set_theme(style="whitegrid")
    plt.plot(epochs_list, acc_sgd, color='#d62728', marker='o', linestyle='--', linewidth=2, label='Standard SGD')
    plt.plot(epochs_list, acc_momentum, color='#2ca02c', marker='D', linewidth=2.5, label='Momentum (γ = 0.9)')
    
    plt.xlabel('Epochs', fontsize=11, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('MLP Classification: SGD vs Momentum Convergence', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(epochs_list)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('mlp_class_comparison.png', dpi=300)
    plt.close()
    
    return acc_sgd, acc_momentum

def compare_mlp_regression(train_x, val_x, train_y, val_y):
    print("--- Comparing SGD vs Momentum (Regression) ---")
    epochs_list = [50, 100, 200, 500]
    
    y_train_reg = train_y.reshape(-1, 1)
    
    lr = 0.001  # Best learning rate identified
    
    mse_sgd = []
    mse_momentum = []
    
    D = train_x.shape[1]
    
    for epochs in epochs_list:
        # SGD (momentum = 0.0)
        mlp_sgd = MLP(dimensions=(D, 64, 1), activations=(ReLU, ReLU))
        mlp_sgd.fit(train_x, y_train_reg, loss=MSE, epochs=epochs, batch_size=32, learning_rate=lr, momentum=0.0)
        preds_sgd = mlp_sgd.predict(val_x).flatten()
        mse_sgd.append(mse_fn(preds_sgd, val_y))
        
        # Momentum (momentum = 0.9)
        mlp_mom = MLP(dimensions=(D, 64, 1), activations=(ReLU, ReLU))
        mlp_mom.fit(train_x, y_train_reg, loss=MSE, epochs=epochs, batch_size=32, learning_rate=lr, momentum=0.9)
        preds_mom = mlp_mom.predict(val_x).flatten()
        mse_momentum.append(mse_fn(preds_mom, val_y))
        
        print(f"Epochs: {epochs:<3d} | SGD MSE: {mse_sgd[-1]:.4f} | Momentum MSE: {mse_momentum[-1]:.4f}")
        
    # Generate line plot
    plt.figure(figsize=(7, 4.5))
    sns.set_theme(style="whitegrid")
    plt.plot(epochs_list, mse_sgd, color='#d62728', marker='o', linestyle='--', linewidth=2, label='Standard SGD')
    plt.plot(epochs_list, mse_momentum, color='#2ca02c', marker='D', linewidth=2.5, label='Momentum (γ = 0.9)')
    
    plt.xlabel('Epochs', fontsize=11, fontweight='bold')
    plt.ylabel('Validation MSE (Lower is Better)', fontsize=11, fontweight='bold')
    plt.title('MLP Regression: SGD vs Momentum Convergence', fontsize=12, fontweight='bold', pad=12)
    plt.xticks(epochs_list)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig('mlp_reg_comparison.png', dpi=300)
    plt.close()
    
    return mse_sgd, mse_momentum

if __name__ == "__main__":
    train_x, val_x, train_y_reg, val_y_reg, train_y_class, val_y_class = load_and_prepare_data()
    
    best_k, kmeans_acc, kmeans_f1 = tune_kmeans(train_x, val_x, train_y_class, val_y_class)
    compare_mlp_classification(train_x, val_x, train_y_class, val_y_class)
    compare_mlp_regression(train_x, val_x, train_y_reg, val_y_reg)
    
    print("\nVisualizations successfully saved as:")
    print(" - kmeans_sweep.png")
    print(" - mlp_class_comparison.png")
    print(" - mlp_reg_comparison.png")
