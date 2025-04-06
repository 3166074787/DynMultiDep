import argparse
import os

import yaml
import torch.nn as nn

import wandb
import torch
from tqdm import tqdm

from models import DynMultiDep
from datasets import get_dvlog_dataloader, get_lmvd_dataloader

CONFIG_PATH = "./config/config.yaml"
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

def parse_args():
    """
    Parse command line arguments and load configuration from YAML file.
    
    Returns:
        Parsed arguments
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )
    # Arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-wdb", "--if_wandb", type=bool)
    parser.add_argument("-tqdm", "--tqdm_able", type=bool)
    parser.add_argument("-tr", "--train", type=bool)
    parser.add_argument("-d", "--device", type=str, nargs="*")
    parser.set_defaults(**config)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    return args


def train_epoch(
        net, train_loader, loss_fn, optimizer, device,
        current_epoch, total_epochs, tqdm_able
):
    """
    Train model for one epoch.
    
    Args:
        net: Model to train
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        current_epoch: Current epoch number
        total_epochs: Total number of epochs
        tqdm_able: Whether to display progress bar
        
    Returns:
        Dictionary with training metrics
    """
    net.train()
    sample_count = 0
    running_loss = 0.
    correct_count = 0

    with tqdm(
            train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
            leave=False, unit="batch", disable=tqdm_able
    ) as pbar:
        for x, y, mask in pbar:
            # Move data to device
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            mask = mask.to(device)

            # Forward pass
            y_pred, additional_loss = net(x, padding_mask=mask)

            # Calculate loss
            loss = loss_fn(y_pred, y.to(torch.float32))
            total_loss = loss + additional_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update metrics
            sample_count += x.shape[0]
            running_loss += total_loss.item() * x.shape[0]
            pred = (y_pred > 0.).int()
            correct_count += (pred == y).sum().item()

            # Update progress bar
            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "acc": correct_count / sample_count,
            })

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(
        net, val_loader, loss_fn, device, tqdm_able
):
    """
    Validate model.
    
    Args:
        net: Model to validate
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to validate on
        tqdm_able: Whether to display progress bar
        
    Returns:
        Dictionary with validation metrics
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
                val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                # Move data to device
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                mask = mask.to(device)

                # Forward pass
                y_pred, additional_loss = net(x, padding_mask=mask)

                # Calculate loss
                loss = loss_fn(y_pred, y.to(torch.float32))
                total_loss = loss + additional_loss

                # Update metrics
                sample_count += x.shape[0]
                running_loss += total_loss.item() * x.shape[0]
                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                # Calculate metrics for progress bar
                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                # Update progress bar
                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    # Calculate final metrics
    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )
    
    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }


def main():
    """
    Main function to run training and evaluation.
    """
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    
    for i_iter in range(1):
        # Initialize wandb if enabled
        if args.if_wandb:
            wandb_run_name = f"{args.model}-{args.train_gender}-{args.test_gender}"
            wandb.init(
                project="mamnba_ad", config=args, name=wandb_run_name,
            )
            args = wandb.config
        print(args)
        
        # Create output directories
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/samples", exist_ok=True)
        os.makedirs(f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints", exist_ok=True)

        # Initialize model
        if args.model == "DynMultiDep":
            if args.dataset == 'lmvd':
                net = DynMultiDep(**args.mmmamba_lmvd)
            elif args.dataset == 'dvlog':
                net = DynMultiDep(**args.mmmamba)
        else:
            raise NotImplementedError(f"The {args.model} method has not been implemented by this repo")

        # Setup device and move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and len(args.device) > 1:
            net = nn.DataParallel(net, device_ids=[0])
        net = net.to(device)

        # Load appropriate dataset
        if args.dataset == 'dvlog':
            train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        elif args.dataset == 'lmvd':
            train_loader = get_lmvd_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_lmvd_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_lmvd_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )

        # Setup loss function and optimizer
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_val_acc = -1.0
        best_test_acc = -1.0
        
        # Training loop
        if args.train:
            for epoch in range(args.epochs):
                # Train for one epoch
                train_results = train_epoch(
                    net, train_loader, loss_fn, optimizer,
                    device, epoch, args.epochs, args.tqdm_able
                )
                
                # Validate
                val_results = val(net, val_loader, loss_fn, device, args.tqdm_able)

                # Calculate combined validation score
                val_acc = (
                    val_results["acc"] + 
                    val_results["precision"] + 
                    val_results["recall"] + 
                    val_results["f1"]
                ) / 4.0
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        net.state_dict(),
                        f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt"
                    )

                # Log metrics to wandb if enabled
                if args.if_wandb:
                    wandb.log({
                        "loss/train": train_results["loss"],
                        "acc/train": train_results["acc"],
                        "loss/val": val_results["loss"],
                        "acc/val": val_results["acc"],
                        "precision/val": val_results["precision"],
                        "recall/val": val_results["recall"],
                        "f1/val": val_results["f1"]
                    })

        # Test best model
        with torch.no_grad():
            # Load best model
            net.load_state_dict(
                torch.load(
                    f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_iter)}/checkpoints/best_model.pt",
                    map_location=device
                )
            )
            net.eval()
            
            # Run test
            test_results = val(net, test_loader, loss_fn, device, args.tqdm_able)
            print("Test results:")
            print(test_results)

            # Save test results
            with open(f'./results/{args.dataset}_{args.model}_{str(i_iter)}.txt', 'w') as f:
                test_result_str = (
                    f'Accuracy:{test_results["acc"]}, '
                    f'Precision:{test_results["precision"]}, '
                    f'Recall:{test_results["recall"]}, '
                    f'F1:{test_results["f1"]}, '
                    f'Avg:{(test_results["acc"] + test_results["precision"] + test_results["recall"] + test_results["f1"]) / 4.0}'
                )
                f.write(test_result_str)

    # Close wandb
    if args.if_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
