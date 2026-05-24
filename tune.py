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
        # fit assigns labels based on voting on training data
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

    # Generate premium plot for K-Means
    plt.figure(figsize=(7, 5))
    sns.set_theme(style="whitegrid")
    
    # Plot accuracy and F1 on same chart (dual axis)
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    
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

def tune_mlp_classification(train_x, val_x, train_y, val_y):
    print("--- Tuning MLP Classification ---")
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 0.5]
    epochs_list = [50, 100, 200, 500]
    
    # Convert labels to one-hot for training MLP
    y_train_oh = np.zeros((train_y.shape[0], 3))
    y_train_oh[np.arange(y_train_oh.shape[0]), train_y.astype(int)] = 1
    
    results_acc = np.zeros((len(learning_rates), len(epochs_list)))
    results_f1 = np.zeros((len(learning_rates), len(epochs_list)))
    
    best_lr = 1e-3
    best_epochs = 100
    best_acc = 0.0
    best_f1 = 0.0
    
    D = train_x.shape[1]
    C = 3
    
    for i, lr in enumerate(learning_rates):
        for j, epochs in enumerate(epochs_list):
            # Using same architecture as main.py (D, 64, C) with (ReLU, Sigmoid)
            mlp = MLP(dimensions=(D, 64, C), activations=(ReLU, Sigmoid))
            mlp.fit(train_x, y_train_oh, loss=CrossEntropy, epochs=epochs, batch_size=32, learning_rate=lr)
            
            preds = mlp.predict(val_x)
            pred_labels = np.argmax(preds, axis=1)
            
            acc = accuracy_fn(pred_labels, val_y)
            f1 = macrof1_fn(pred_labels, val_y)
            
            results_acc[i, j] = acc
            results_f1[i, j] = f1
            
            print(f"lr={lr:<5g} | epochs={epochs:<3d} | Val Accuracy: {acc:6.2f}% | Val Macro F1: {f1:6.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_epochs = epochs
                best_f1 = f1
                
    print(f"Best MLP Classification: lr={best_lr}, epochs={best_epochs} with Accuracy={best_acc:.2f}%, F1={best_f1:.4f}\n")
    
    # Generate Heatmap
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    ax = sns.heatmap(results_acc, annot=True, fmt=".2f", cmap="viridis",
                     xticklabels=epochs_list, yticklabels=learning_rates,
                     cbar_kws={'label': 'Validation Accuracy (%)'})
    ax.set_xlabel('Epochs', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('MLP Classification: Validation Accuracy Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('mlp_class_heatmap.png', dpi=300)
    plt.close()
    
    return best_lr, best_epochs, best_acc, best_f1

def tune_mlp_regression(train_x, val_x, train_y, val_y):
    print("--- Tuning MLP Regression ---")
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epochs_list = [50, 100, 200, 500]
    
    y_train_reg = train_y.reshape(-1, 1)
    
    results_mse = np.zeros((len(learning_rates), len(epochs_list)))
    
    best_lr = 1e-5
    best_epochs = 100
    best_mse = float('inf')
    
    D = train_x.shape[1]
    
    for i, lr in enumerate(learning_rates):
        for j, epochs in enumerate(epochs_list):
            # Using same architecture as main.py (D, 64, 1) with (ReLU, ReLU)
            mlp = MLP(dimensions=(D, 64, 1), activations=(ReLU, ReLU))
            mlp.fit(train_x, y_train_reg, loss=MSE, epochs=epochs, batch_size=32, learning_rate=lr)
            
            preds = mlp.predict(val_x)
            pred_labels = preds.flatten()
            
            mse = mse_fn(pred_labels, val_y)
            results_mse[i, j] = mse
            
            print(f"lr={lr:<5g} | epochs={epochs:<3d} | Val MSE: {mse:6.4f}")
            
            if mse < best_mse:
                best_mse = mse
                best_lr = lr
                best_epochs = epochs
                
    print(f"Best MLP Regression: lr={best_lr}, epochs={best_epochs} with MSE={best_mse:.4f}\n")
    
    # Generate Heatmap (using inverse colormap because lower MSE is better)
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    ax = sns.heatmap(results_mse, annot=True, fmt=".4f", cmap="viridis_r",
                     xticklabels=epochs_list, yticklabels=learning_rates,
                     cbar_kws={'label': 'Validation MSE'})
    ax.set_xlabel('Epochs', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('MLP Regression: Validation MSE Heatmap', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('mlp_reg_heatmap.png', dpi=300)
    plt.close()
    
    return best_lr, best_epochs, best_mse

if __name__ == "__main__":
    train_x, val_x, train_y_reg, val_y_reg, train_y_class, val_y_class = load_and_prepare_data()
    
    best_k, kmeans_acc, kmeans_f1 = tune_kmeans(train_x, val_x, train_y_class, val_y_class)
    best_lr_c, best_ep_c, mlp_class_acc, mlp_class_f1 = tune_mlp_classification(train_x, val_x, train_y_class, val_y_class)
    best_lr_r, best_ep_r, mlp_reg_mse = tune_mlp_regression(train_x, val_x, train_y_reg, val_y_reg)
    
    print("================Summary================")
    print(f"K-Means Classification (K={best_k}): Accuracy={kmeans_acc:.2f}%, F1={kmeans_f1:.4f}")
    print(f"MLP Classification (lr={best_lr_c}, epochs={best_ep_c}): Accuracy={mlp_class_acc:.2f}%, F1={mlp_class_f1:.4f}")
    print(f"MLP Regression (lr={best_lr_r}, epochs={best_ep_r}): MSE={mlp_reg_mse:.4f}")
    print("=======================================")
