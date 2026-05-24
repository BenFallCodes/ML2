import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import Sigmoid, ReLU
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        n_train = int(0.8 * train_features.shape[0])
        test_features = train_features[n_train:]
        test_labels_reg = train_labels_reg[n_train:]
        test_labels_classif = train_labels_classif[n_train:]
        train_features = train_features[:n_train]
        train_labels_reg = train_labels_reg[:n_train]
        train_labels_classif = train_labels_classif[:n_train]

    # Normalize features
    means = np.mean(train_features, axis=0, keepdims=True)
    stds = np.std(train_features, axis=0, keepdims=True)
    stds[stds == 0] = 1.0 # prevent division by zero
    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        D = train_features.shape[1]
        if args.task == "classification":
            C = 3
            method_obj = MLP(dimensions=(D, 64, C), activations=(ReLU, Sigmoid))
        else:
            method_obj = MLP(dimensions=(D, 64, 1), activations=(ReLU, ReLU))
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        if args.method == "mlp":
            y_train = np.zeros((train_labels_classif.shape[0], 3))
            y_train[np.arange(y_train.shape[0]), train_labels_classif.astype(int)] = 1
            method_obj.fit(train_features, y_train, loss=CrossEntropy, epochs=args.max_iters, batch_size=32, learning_rate=args.lr)
            preds = method_obj.predict(test_features)
            pred_labels = np.argmax(preds, axis=1)
        else:
            method_obj.fit(train_features, train_labels_classif)
            pred_labels = method_obj.predict(test_features)
        
        acc = accuracy_fn(pred_labels, test_labels_classif)
        macrof1 = macrof1_fn(pred_labels, test_labels_classif)
        print(f"Accuracy: {acc:.2f}%")
        print(f"Macro F1: {macrof1:.4f}")

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"

        if args.method == "mlp":
            y_train = train_labels_reg.reshape(-1, 1)
            method_obj.fit(train_features, y_train, loss=MSE, epochs=args.max_iters, batch_size=32, learning_rate=args.lr)
            preds = method_obj.predict(test_features)
            pred_labels = preds.flatten()
        else:
            method_obj.fit(train_features, train_labels_reg)
            pred_labels = method_obj.predict(test_features)
        
        mse = mse_fn(pred_labels, test_labels_reg)
        print(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
