import os
import sys
import json
import copy
import torch
import pandas as pd
from argparse import ArgumentParser, ArgumentTypeError

import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import *
from models import MultiClassLogisticRegression, PosthocHybridCBM

import random
random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)

def str2bool(v):
    # argparse's type=bool treats any non-empty string (including "False") as True
    if isinstance(v, bool): return v
    if v.lower() in ("true", "1", "yes", "y"): return True
    if v.lower() in ("false", "0", "no", "n"): return False
    raise ArgumentTypeError(f"Boolean value expected, got {v}")


def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, patience=30, prior_weight=1.0):
    """
    Enhanced training with early stopping and learning rate scheduling.
    patience: number of epochs with no improvement before stopping
    """
    best_val_acc = -float("inf")
    best_model = None
    patience_counter = 0
    
    # Learning rate scheduler: reduce LR if validation accuracy plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long())

            if model.apply_prior != False:
                prior_loss = compute_prior_loss(model)
                loss += prior_weight * prior_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # evaluate the model
        val_acc = evaluate_model(model, val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Acc: {val_acc:.2f}")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break
            
        # Adaptive learning rate scheduling
        scheduler.step(val_acc)
    
    return best_model


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    
    return accuracy


def get_results(args, label2index, classifier_list, df_train_log, y_train, df_val_log, y_val, df_ood_log, y_ood):
    batch_size, learning_rate, num_epochs = args.batch_size, args.lr, args.epochs
    # Convert features and labels to PyTorch tensors
    X_train_torch = torch.tensor(df_train_log.values).float().to(device)
    y_train_torch = torch.tensor(y_train).to(device)
    X_val_torch = torch.tensor(df_val_log.values).float().to(device)
    y_val_torch = torch.tensor(y_val).to(device)
    X_ood_torch = torch.tensor(df_ood_log.values).float().to(device)
    y_ood_torch = torch.tensor(y_ood).to(device)

    # Create DataLoader instances with optimized batch size for better convergence
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)
    ood_dataset = TensorDataset(X_ood_torch, y_ood_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(torch.unique(y_train_torch))  # Assuming y_train contains all classes
    class_names = list(label2index.keys())
    concepts = list(classifier_list.keys())

    # Get the prior matrix
    prior = get_prior_matrix(args.modality, class_names, concepts)

    # Define the logistic regression model
    if args.mode == "pcbm":
        model = PosthocHybridCBM(n_concepts=len(concepts), 
                                 n_classes=num_classes, 
                                 n_image_features=X_train_torch.shape[1] - len(concepts))
    else:
        model = MultiClassLogisticRegression(num_features=X_train_torch.shape[1], 
                                             num_classes=num_classes, 
                                             prior=prior,
                                             apply_prior=args.add_prior)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    # Train the model with enhanced training
    best_model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience=args.patience, prior_weight=args.prior_weight)

    # Evaluate the model
    val_acc = evaluate_model(best_model, val_loader)
    ood_acc = evaluate_model(best_model, ood_loader)

    average_acc = round((val_acc + ood_acc) / 2, 2)
    gap = round(abs(val_acc - ood_acc), 2)
    
    return val_acc, ood_acc, gap, average_acc, best_model


def run_classification(args, dataset_name, classifier_list, clip_model, tokenizer, preprocess, random_seed=42):
    # Load the features
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_path = os.path.join(workspace_root, f"data/features/{args.model_name}/{dataset_name}_label.pt")
    feature_path = os.path.join(workspace_root, f"data/features/{args.model_name}/{dataset_name}")
    label2index = torch.load(label_path, weights_only=False)
    X_train, y_train, X_val, y_val, X_ood, y_ood = load_features(feature_path, label2index, args.shots, args.normalize, random_seed)

    if args.mode == "binary":
        df_train_log, df_val_log, df_ood_log = binary_features(X_train, X_val, X_ood, classifier_list)
    elif args.mode == "linear_probe":
        df_train_log, df_val_log, df_ood_log = linear_features(X_train, X_val, X_ood, args.number_of_features)
    elif args.mode == "dot_product":
        df_train_log, df_val_log, df_ood_log = dot_product_features(X_train, X_val, X_ood, classifier_list, clip_model, tokenizer)
    elif args.mode == "pcbm":
        df_train_log, df_val_log, df_ood_log = pcbm_features(X_train, X_val, X_ood, classifier_list, clip_model, tokenizer, preprocess, args.number_of_features)

    print("Train size: ", df_train_log.shape, "Test size: ", df_val_log.shape, "OOD size: ", df_ood_log.shape)

    val_acc, ood_acc, gap, average_acc, best_model = get_results(args, label2index, classifier_list, df_train_log, y_train, df_val_log, y_val, df_ood_log, y_ood)

    print(f"Dataset: {dataset_name}, Mode: {args.mode}", f"Shots: {args.shots}", f"Model: {args.model_name}")
    print(f"Ind Acc: {val_acc}, OOD Acc: {ood_acc}, Gap: {gap}, Average: {average_acc}")
    number_of_features_actual = df_train_log.shape[1]

    return val_acc, ood_acc, gap, average_acc, number_of_features_actual


def run_all_datasets(args, dataset_lists, classifier_list, clip_model, tokenizer, preprocess):
    results_dict = {}
    for dataset_name in dataset_lists:
        ind_acc, out_acc, gap, avg, number_of_features_actual = run_classification(args, dataset_name, classifier_list, clip_model, tokenizer, preprocess)
        results_dict[dataset_name] = {"ind_acc": ind_acc, "out_acc": out_acc, "gap": gap, "avg": avg}

    # reshape to one row
    csv_df = pd.DataFrame.from_dict(results_dict).T
    # Reshape the data
    reshaped_df = pd.DataFrame(csv_df.values.flatten()).T

    # Create new column names
    new_columns = [f"{row_label}_{col_label}" for row_label in csv_df.index for col_label in csv_df.columns]

    # Assign new column names to reshaped dataframe
    reshaped_df.columns = new_columns
    
    # save as csv use all arguments as file name
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_name = os.path.join(workspace_root, f"data/results/{args.modality}_{args.mode}_{args.model_name}_{args.bottleneck}_{args.shots}_{number_of_features_actual}_{args.save_suffix}.csv")

    if args.add_prior: file_name = file_name.replace(".csv", "_prior.csv")
    
    # creat folder if not exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    reshaped_df.to_csv(file_name)
    
    # Print final summary statistics across all datasets
    print("\n" + "="*90)
    print("FINAL ACCURACY SUMMARY - ALL DATASETS")
    print("="*90)
    print(f"Modality: {args.modality} | Model: {args.model_name} | Mode: {args.mode}")
    print(f"Bottleneck: {args.bottleneck} | Features: {number_of_features_actual} | Prior: {args.add_prior}")
    print(f"Hyperparameters:")
    print(f"  • Epochs: {args.epochs} (early stopping, patience={args.patience})")
    print(f"  • Learning Rate: {args.lr} (AdamW, weight_decay={args.weight_decay}, ReduceLROnPlateau)")
    print(f"  • Batch Size: {args.batch_size}")
    print(f"  • Prior Weight: {args.prior_weight if args.add_prior else 'n/a'}")
    print("-"*90)
    print(f"{'Dataset':<20} {'Ind Acc %':<14} {'OOD Acc %':<14} {'Gap %':<14} {'Average %':<14}")
    print("-"*90)
    
    all_ind = []
    all_ood = []
    all_gaps = []
    all_avgs = []
    
    for dataset_name in sorted(results_dict.keys()):
        res = results_dict[dataset_name]
        all_ind.append(res["ind_acc"])
        all_ood.append(res["out_acc"])
        all_gaps.append(res["gap"])
        all_avgs.append(res["avg"])
        print(f"{dataset_name:<20} {res['ind_acc']:<14.2f} {res['out_acc']:<14.2f} {res['gap']:<14.2f} {res['avg']:<14.2f}")
    
    print("-"*90)
    mean_ind = sum(all_ind)/len(all_ind)
    mean_ood = sum(all_ood)/len(all_ood)
    mean_gap = sum(all_gaps)/len(all_gaps)
    mean_avg = sum(all_avgs)/len(all_avgs)
    print(f"{'MEAN':<20} {mean_ind:<14.2f} {mean_ood:<14.2f} {mean_gap:<14.2f} {mean_avg:<14.2f}")
    print(f"{'MAX':<20} {max(all_ind):<14.2f} {max(all_ood):<14.2f} {max(all_gaps):<14.2f} {max(all_avgs):<14.2f}")
    print(f"{'MIN':<20} {min(all_ind):<14.2f} {min(all_ood):<14.2f} {min(all_gaps):<14.2f} {min(all_avgs):<14.2f}")
    print(f"{'STD DEV':<20} {pd.Series(all_ind).std():<14.2f} {pd.Series(all_ood).std():<14.2f} {pd.Series(all_gaps).std():<14.2f} {pd.Series(all_avgs).std():<14.2f}")
    print("="*90)
    print(f"✓ Results saved to: {file_name}")
    print("="*90 + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="binary", help="binary, linear_probe, dot_product")
    parser.add_argument("--bottleneck", type=str, default="PubMed", help="PubMed, prompt, Textbooks")
    parser.add_argument("--shots", type=str, default="all", help="all, 1, 2, 4, 8, 16, 32, 64")
    parser.add_argument("--model_name", type=str, default="whyxrayclip", help="whyxrayclip, whylesionclip")
    parser.add_argument("--acc_threshold", type=float, default=0, help="accuracy threshold for loading classifiers")
    parser.add_argument("--number_of_features", type=int, default=768, help="number of features/concepts")
    parser.add_argument("--normalize", type=str2bool, default=True, help="normalize the features")
    parser.add_argument("--modality", type=str, default="xray", help="xray, natural, skin")
    parser.add_argument("--input_dim", type=int, default=768, help="input dimension for the binary classifiers")
    parser.add_argument("--add_prior", type=str2bool, default=False, help="add prior to the model")
    parser.add_argument("--save_suffix", type=str, default="", help="add suffix to the save folder")
    parser.add_argument("--epochs", type=int, default=400, help="max training epochs")
    parser.add_argument("--patience", type=int, default=40, help="early stopping patience (epochs)")
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--prior_weight", type=float, default=0.5, help="weight of the prior loss (paper uses 1.0)")

    args = parser.parse_args()

    # The CLIP text encoder is only needed for dot_product/pcbm; skip the slow model load otherwise
    if args.mode in ("dot_product", "pcbm"):
        clip_model, tokenizer, preprocess = load_clip_model(args.model_name)
    else:
        clip_model, tokenizer, preprocess = None, None, None

    # Load classifiers
    classifier_list = load_classifier_list(args)
    
    if len(classifier_list) == 0:
        print(f"WARNING: No classifiers loaded! Check your bottleneck file and grounding functions directory.")
        print(f"Bottleneck: {args.bottleneck}, Modality: {args.modality}")
        exit(1)
    
    binary_accuracies = [classifier_list[k][1] for k in classifier_list.keys()]
    print(f"Number of classifiers: {len(classifier_list)}, Mean Acc: {round(sum(binary_accuracies) / len(binary_accuracies), 5)}")

    # Load datasets
    if args.modality == "xray":
        dataset_lists = ["NIH-sex", "NIH-age", "NIH-pos", "CheXpert-race", "NIH-CheXpert", "pneumonia", "COVID-QU", "NIH-CXR", "open-i", "vindr-cxr"]
    elif args.modality == "skin":
        dataset_lists = ["ISIC-sex", "ISIC-age", "ISIC-site", "ISIC-color", "ISIC-hospital", "HAM10000", "BCN20000", "PAD-UFES-20", "Melanoma", "UWaterloo"]
        
    run_all_datasets(args, dataset_lists, classifier_list, clip_model, tokenizer, preprocess)
