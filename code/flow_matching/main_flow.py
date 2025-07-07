import os
import sys
import time
import glob
import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# from IPython.display import clear_output, display

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

global_start_time = time.time()

# from rich.live import Live
# from rich.console import Console
# from rich.table import Table
# from rich.panel import Panel
# import asciichartpy
# from rich.plot import Plot
# from rich.plot import Series
# from rich import box

from emb import *

from fmm import *

# from args import get_arg_parser
import wandb

# Initialize wandb at the start of your script (after your imports)
wandb.login(key="84987f1aa9485c5bac65fe4db16018b4f5a9755e")
wandb.init(
    project="selex-cross-attention",
    config={
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "m": args.m,
        "num_splits": args.num_splits,
        "num_epochs": args.num_epochs,
        "optimizer": "Adam",
        "lr": 1e-3,
    },
    name="Flow Matching - Draft 1[with Lookup Embedding using NN]"#f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

class SimpleTestDataset(torch.utils.data.Dataset):
    def __init__(self, data, file_indices):
        self.data = data
        self.file_indices = file_indices
        _, _, self.rho, self.rho_next = get_counts_and_abundance(self.data, sorted_keys[file_indices[0]], sorted_keys[file_indices[0]+1])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.file_indices[idx], self.rho[idx], self.rho_next[idx]

class SimpleTrainDataset(Dataset):
    def __init__(self, data, file_indices):
        self.data = data
        self.file_indices = file_indices
        _, _, self.rho, self.rho_next = get_counts_and_abundance(self.data, sorted_keys[file_indices[0]], sorted_keys[file_indices[0]+1])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.file_indices[idx], self.rho[idx], self.rho_next[idx]

class SimpleValDataset(Dataset):
    def __init__(self, data, file_indices):
        self.data = data
        self.file_indices = file_indices
        _, _, self.rho, self.rho_next = get_counts_and_abundance(self.data, sorted_keys[file_indices[0]], sorted_keys[file_indices[0]+1])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.file_indices[idx], self.rho[idx], self.rho_next[idx]
    
embed_dim = args.embedding_dim
num_heads = args.num_heads
batch_size = args.batch_size
m = args.m
k = args.num_splits
num_epochs = args.num_epochs
plot_base = args.plot_base
model_dir = args.model_dir

# Check if CUDA is available and set the device accordingly
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Initialize the network, conditional velocity field, and OTFlowMatching model
n_freqs = args.n_freqs
round_embedding_dim = args.round_embedding_dim
net = Net(in_dim=embed_dim, out_dim = embed_dim, h_dims=[512]*5, n_freqs=n_freqs, round_embedding_dim=round_embedding_dim).to(device)
v_t = CondVF(net).to(device)
model = OTFlowMatching(sig_min=0.001)

# Optimizer for the network
optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)

num_params = sum(p.numel() for p in v_t.parameters())
trainable_params = sum(p.numel() for p in v_t.parameters() if p.requires_grad)
non_trainable_params = num_params - trainable_params
print(f"""Number of parameters : {num_params}, Number of trainable parameters : {trainable_params}
       and Number of non-trainable params : {non_trainable_params}""")

excluded_file = sorted_keys[-1] if len(sorted_keys) > 2 else None
all_files = sorted_keys[:-1]#[f for f in sorted_keys if f != excluded_file]
file_indices = torch.tensor(flat_dataset.file_indices)

# Split the dataset into train, validation, and test sets in the ratio 80:10:10
num_files = len(all_files)
num_train = int(train_prop * num_files)
num_val = int(val_prop * num_files)
num_test = num_files - num_train - num_val

# Set aside the test set once and keep it fixed
files_shuffled = all_files[:]
files_shuffled = [files_shuffled[i] for i in torch.randperm(len(files_shuffled))]
test = set(files_shuffled[-num_test:])
remaining_files = [f for f in files_shuffled if f not in test]

test_indices = torch.tensor([flat_dataset.keys.index(f) for f in test])

def save_dataset_object(dataset_obj, split_name, split_idx):
    save_dir = f"scratch/datasets/split{split_idx+1}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(dataset_obj, os.path.join(save_dir, f"{split_name}_dataset_obj.pt"))

def load_dataset_object(split_name, split_idx):
    load_path = f"scratch/datasets/split{split_idx+1}/{split_name}_dataset_obj.pt"
    if os.path.exists(load_path):
        return torch.load(load_path, weights_only=False)
    return None

def save_test_dataset(test_dataset):
    save_dir = "scratch/datasets"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(test_dataset, os.path.join(save_dir, "test_dataset.pt"))

def load_test_dataset():
    load_path = "scratch/datasets/test_dataset.pt"
    if os.path.exists(load_path):
        return torch.load(load_path, weights_only=False)
    return None

test_dataset = load_test_dataset()
if test_dataset is None:
    test_batches = []
    for batch, batch_file_indices in tqdm(dataloader, desc="Test dataloader under progress"):
        batch_file_indices_tensor = torch.as_tensor(batch_file_indices)
        mask = torch.isin(batch_file_indices_tensor, test_indices)
        if mask.any():
            selected_indices = torch.nonzero(mask, as_tuple=True)[0]
            selected_batch = batch[selected_indices]
            selected_file_indices = batch_file_indices_tensor[selected_indices]
            test_batches.append((selected_batch, selected_file_indices))

    if test_batches:
        test_data = torch.cat([b for b, _ in test_batches], dim=0)
        test_file_indices = torch.cat([idx for _, idx in test_batches], dim=0)
        print("Creating test dataset")
        start_test_dataset_time = time.time()
        test_dataset = SimpleTestDataset(test_data, test_file_indices)
        elapsed_test_dataset_time = time.time() - start_test_dataset_time
        print(f"Test dataset created in {elapsed_test_dataset_time:.2f} seconds")
        save_test_dataset(test_dataset)
        print("Test dataset saved")
    else:
        test_dataset = None

if test_dataset is not None:
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Test dataloader created")
else:
    test_dataloader = None

# For k folds, shuffle and split remaining files into train/val
splits = []
for _ in range(k):
    rem_shuffled = remaining_files[:]
    rem_shuffled = [rem_shuffled[i] for i in torch.randperm(len(rem_shuffled))]
    train = set(rem_shuffled[:num_train])
    val = set(rem_shuffled[num_train:num_train + num_val])
    splits.append((train, val))

for split_idx, (train_files, val_files) in enumerate(splits):
    train_indices = torch.tensor([flat_dataset.keys.index(f) for f in train_files])
    val_indices = torch.tensor([flat_dataset.keys.index(f) for f in val_files])

    train_batches = []
    val_batches = []
    with tqdm(dataloader, desc="Train and Validation", leave=False) as pbar:
        for batch, batch_file_indices in pbar:
            batch.to(device)
            batch_file_indices_tensor = torch.as_tensor(batch_file_indices)
            # Mask for train
            train_mask = torch.isin(batch_file_indices_tensor, train_indices)
            if train_mask.any():
                selected_indices = torch.nonzero(train_mask, as_tuple=True)[0]
                selected_batch = batch[selected_indices]
                selected_file_indices = batch_file_indices_tensor[selected_indices]
                train_batches.append((selected_batch, selected_file_indices))
            # Mask for val
            val_mask = torch.isin(batch_file_indices_tensor, val_indices)
            if val_mask.any():
                selected_indices = torch.nonzero(val_mask, as_tuple=True)[0]
                selected_batch = batch[selected_indices]
                selected_file_indices = batch_file_indices_tensor[selected_indices]
                val_batches.append((selected_batch, selected_file_indices))

    # Train dataset
    train_dataset_masked = load_dataset_object("train", split_idx)
    if train_dataset_masked is None:
        if train_batches:
            train_data = torch.cat([b for b, _ in train_batches], dim=0)
            train_file_indices_flat = torch.cat([idx for _, idx in train_batches], dim=0)
            print("Creating train dataset")
            start_train_dataset_time = time.time()
            train_dataset_masked = SimpleTrainDataset(train_data, train_file_indices_flat)
            elapsed_train_dataset_time = time.time() - start_train_dataset_time
            print(f"Train dataset created in {elapsed_train_dataset_time:.2f} seconds")
            save_dataset_object(train_dataset_masked, "train", split_idx)
            print("Train dataset object saved")
        else:
            train_dataset_masked = None

    if train_dataset_masked is not None:
        train_dataloader = DataLoader(train_dataset_masked, batch_size=batch_size, shuffle=not True)
        print("Training Dataloader created")
    else:
        train_dataloader = None
        print("No training batches found for this split.")

    # Validation dataset
    val_dataset_masked = load_dataset_object("val", split_idx)
    if val_dataset_masked is None:
        if val_batches:
            val_data = torch.cat([b for b, _ in val_batches], dim=0)
            val_file_indices_flat = torch.cat([idx for _, idx in val_batches], dim=0)
            print("Creating validation dataset")
            start_val_dataset_time = time.time()
            val_dataset_masked = SimpleValDataset(val_data, val_file_indices_flat)
            elapsed_val_dataset_time = time.time() - start_val_dataset_time
            print(f"Validation dataset created in {elapsed_val_dataset_time:.2f} seconds")
            save_dataset_object(val_dataset_masked, "val", split_idx)
            print("Validation dataset object saved")
        else:
            val_dataset_masked = None

    if val_dataset_masked is not None:
        val_dataloader = DataLoader(val_dataset_masked, batch_size=batch_size, shuffle=False)
        print("Validation dataloader created")
    else:
        val_dataloader = None
        print("No validation batches found for this split.")
    
    print(f"Split {split_idx+1} : Dataloaders - Train, Validation and 'Test'* - all created.")

    split_train_losses = []
    split_val_losses = []
    split_train_abs_rel_errors = []
    split_val_abs_rel_errors = []
    split_train_log_losses = []
    split_val_log_losses = []

    split_start_time = time.time()
    for epoch in range(num_epochs):
        v_t.train()
        train_losses = []
        train_abs_rel_errors = []
        train_log_losses = []
        epoch_start_time = time.time()
        batch_times = []

        # with Live(generate_layout(), refresh_per_second=4, screen=False) as live:
        with tqdm(train_dataloader, desc=f"Split {split_idx+1} Epoch {epoch+1}/{num_epochs} [Train]", leave=not False) as pbar:
            for batch, file_indices, rho, rho_next in pbar:
                if batch.shape[0] != batch_size:
                    print(f"Skipping batch with shape {batch.shape} (expected {batch_size})")
                    continue
                batch_start_time = time.time()
                filename = flat_dataset.keys[file_indices[0].item()]
                if filename == excluded_file:
                    continue
                if batch.shape[0] != batch_size:
                    print(f"Skipping batch with shape {batch.shape} (expected {batch_size})")
                    continue

                batch = batch.to(device)
                # print(batch[:5])

                embeddings = embedding_layer(batch)
                pos_emb = pos_embedding_layer(batch)
                final_embedding = embeddings + pos_emb

                next_file_idx = file_indices[0].item() + 1
                # Find indices where file_indices == next_file_idx
                candidate_indices = torch.where(torch.tensor(flat_dataset.file_indices) == next_file_idx)[0]
                # print(len(candidate_indices))
                sampled_indices = candidate_indices[torch.randint(0, len(candidate_indices), (batch_size,))]
                next_file_batch = torch.stack(flat_dataset.data).to(device)
                next_file_batch = next_file_batch[sampled_indices]

                # print(next_file_batch[:5])

                embeddings_tgt = embedding_layer(next_file_batch)
                pos_emb_tgt = pos_embedding_layer(next_file_batch)
                final_embedding_tgt = embeddings_tgt + pos_emb_tgt

                file_indices = file_indices.to(device)

                loss = model.loss(v_t, final_embedding, final_embedding_tgt, file_indices)

                if torch.isnan(loss):
                    # print("Loss is NaN, skipping this batch.")
                    continue
                if torch.isinf(loss):
                    # print("Loss is Inf, skipping this batch.")
                    continue
                if loss < 0:
                    # print("Loss is negative, resetting to zero.")
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                if loss > 1000:
                    # print("Loss is too high, resetting to 1000.")
                    loss = torch.tensor(1000.0, device=device, requires_grad=True)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                with torch.no_grad():
                    y_pred = v_t.decode(final_embedding, file_indices)
                y_true = final_embedding_tgt

                # print(loss, y_pred.shape, y_true.shape)

                # Absolute relative error
                abs_rel_error = torch.median(torch.abs((y_pred - y_true) / (y_true + 1e-8)))
                train_abs_rel_errors.append(abs_rel_error.detach().cpu())

                # Log of the loss
                train_log_losses.append(torch.log(torch.tensor(loss.item() + 1e-8)).item())

                # print(f"Loss : {train_losses}, Abs rel error : {train_abs_rel_errors}, Log loss : {train_log_losses}")
                # exit(0)

                # live.update(generate_layout())
                # time.sleep(0.1)

                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                # pbar.set_postfix({f"Truth": f"{y_true.item():.3f}", "Prediction": f"{y_pred.item():.3f}","Batch Loss": loss.item(), "Batch Time (s)": f"{batch_time:.3f}"})
                pbar.set_postfix({f"Batch Loss": loss.item(), "Batch Time (s)": f"{batch_time:.3f}"})
                # Log batch-level losses
                wandb.log({
                    "train/loss": loss.item(),
                    "train/abs_rel_error": abs_rel_error.mean().item(),
                    "train/log_loss": train_log_losses[-1]
                })

        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        epoch_time = time.time() - epoch_start_time
        print(f"Split {split_idx+1} | Epoch {epoch+1}/{num_epochs} | Avg Batch Time: {avg_batch_time:.3f}s | Epoch Time: {epoch_time:.2f}s")

        v_t.eval()
        val_losses = []
        val_abs_rel_errors = []
        val_log_losses = []
        val_batch_times = []
        global_step = (split_idx * num_epochs + epoch + 1) * len(train_dataloader)
        with torch.no_grad():
            with tqdm(val_dataloader, desc=f"Split {split_idx+1} Epoch {epoch+1}/{num_epochs} [Val]", leave=False) as pbar:
                for batch, file_indices, rho, rho_next in pbar:
                    val_batch_start = time.time()
                    filename = flat_dataset.keys[file_indices[0].item()]
                    if filename == excluded_file:
                        continue

                    batch = batch.to(device)
                    # print(batch[:5])

                    embeddings = embedding_layer(batch)
                    pos_emb = pos_embedding_layer(batch)
                    final_embedding = embeddings + pos_emb

                    next_file_idx = file_indices[0].item() + 1
                    # Find indices where file_indices == next_file_idx
                    candidate_indices = torch.where(torch.tensor(flat_dataset.file_indices) == next_file_idx)[0]
                    # print(len(candidate_indices))
                    sampled_indices = candidate_indices[torch.randint(0, len(candidate_indices), (batch_size,))]
                    next_file_batch = torch.stack(flat_dataset.data).to(device)
                    next_file_batch = next_file_batch[sampled_indices]

                    # print(next_file_batch[:5])

                    embeddings_tgt = embedding_layer(next_file_batch)
                    pos_emb_tgt = pos_embedding_layer(next_file_batch)
                    final_embedding_tgt = embeddings_tgt + pos_emb_tgt

                    file_indices = file_indices.to(device)

                    loss = model.loss(v_t, final_embedding, final_embedding_tgt, file_indices)

                    if torch.isnan(loss):
                        # print("Loss is NaN, skipping this batch.")
                        continue
                    if torch.isinf(loss):
                        # print("Loss is Inf, skipping this batch.")
                        continue
                    if loss < 0:
                        # print("Loss is negative, resetting to zero.")
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
                    if loss > 1000:
                        # print("Loss is too high, resetting to 1000.")
                        loss = torch.tensor(1000.0, device=device, requires_grad=True)

                    val_losses.append(loss.item())

                    y_pred = v_t.decode(final_embedding, file_indices)
                    y_true = final_embedding_tgt

                    # Absolute relative error
                    abs_rel_error = torch.abs((y_pred - y_true) / (y_true + 1e-8))
                    val_abs_rel_errors.append(abs_rel_error.detach().cpu())

                    # Log loss
                    val_log_losses.append(torch.log(torch.tensor(loss.item() + 1e-8)).item())

                    val_batch_time = time.time() - val_batch_start
                    val_batch_times.append(val_batch_time)
                    pbar.set_postfix({"Val Batch Loss": loss.item(), "Batch Time (s)": f"{val_batch_time:.3f}"})
                    
                    wandb.log({
                        "val/loss": loss.item(),
                        "val/abs_rel_error": abs_rel_error.mean().item(),
                        "val/log_loss": val_log_losses[-1]
                    })
        

        avg_val_batch_time = sum(val_batch_times) / len(val_batch_times) if val_batch_times else 0
        print(f"Split {split_idx+1} | Epoch {epoch+1}/{num_epochs} | Avg Val Batch Time: {avg_val_batch_time:.3f}s")

        split_train_losses.append(sum(train_losses)/len(train_losses) if train_losses else 0)
        split_val_losses.append(sum(val_losses)/len(val_losses) if val_losses else 0)

        # Aggregate and store mean absolute relative error for this epoch
        if train_abs_rel_errors:
            train_abs_rel_errors_cat = torch.cat(train_abs_rel_errors)
            train_abs_rel_errors_cat = train_abs_rel_errors_cat[~torch.isnan(train_abs_rel_errors_cat)]
            train_abs_rel_errors_cat = train_abs_rel_errors_cat[~torch.isinf(train_abs_rel_errors_cat)]
            split_train_abs_rel_errors.append(train_abs_rel_errors_cat.median().item())
            # print(split_train_abs_rel_errors)
        else:
            split_train_abs_rel_errors.append(0.0)
        if val_abs_rel_errors:
            val_abs_rel_errors_cat = torch.cat(val_abs_rel_errors)
            val_abs_rel_errors_cat = val_abs_rel_errors_cat[~torch.isnan(val_abs_rel_errors_cat)]
            val_abs_rel_errors_cat = val_abs_rel_errors_cat[~torch.isinf(val_abs_rel_errors_cat)]
            split_val_abs_rel_errors.append(val_abs_rel_errors_cat.median().item())
            # print(split_val_abs_rel_errors)
        else:
            split_val_abs_rel_errors.append(0.0)

        # Aggregate and store mean log loss for this epoch
        split_train_log_losses.append(sum(train_log_losses)/len(train_log_losses) if train_log_losses else 0)
        split_val_log_losses.append(sum(val_log_losses)/len(val_log_losses) if val_log_losses else 0)

        # Save the model after every epoch
        save_dir = model_dir+f"/split{split_idx+1}/epoch{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(v_t.state_dict(), os.path.join(save_dir, f"model_split_{split_idx+1}_epoch_{epoch+1}.pth"))
        if (epoch + 1) % 20 == 0 or not epoch:
            print(f"Split {split_idx+1} | Epoch {epoch+1}/{num_epochs} | Train Loss: {split_train_losses[-1]:.4f} | Val Loss: {split_val_losses[-1]:.4f}")
            print(f"Split {split_idx+1} | Epoch {epoch+1}/{num_epochs} | Train Abs Rel Error: {split_train_abs_rel_errors[-1]:.4f} | Val Abs Rel Error: {split_val_abs_rel_errors[-1]:.4f}")
            print(f"Split {split_idx+1} | Epoch {epoch+1}/{num_epochs} | Train Log Loss: {split_train_log_losses[-1]:.4f} | Val Log Loss: {split_val_log_losses[-1]:.4f}")

            # Plot and save loss, error, and log loss curves after every 20 epochs
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(range(1, len(split_train_losses)+1), split_train_losses, label='Train Loss')
            plt.plot(range(1, len(split_val_losses)+1), split_val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Split {split_idx+1} Loss Curves')
            plt.legend()
            plt.subplot(1, 3, 2)
            plt.plot(range(1, len(split_train_abs_rel_errors)+1), split_train_abs_rel_errors, label='Train Abs Rel Error')
            plt.plot(range(1, len(split_val_abs_rel_errors)+1), split_val_abs_rel_errors, label='Validation Abs Rel Error')
            plt.xlabel('Epoch')
            plt.ylabel('Abs Rel Error')
            plt.title(f'Split {split_idx+1} Abs Rel Error Curves')
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.plot(range(1, len(split_train_log_losses)+1), split_train_log_losses, label='Train Log Loss')
            plt.plot(range(1, len(split_val_log_losses)+1), split_val_log_losses, label='Validation Log Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Log Loss')
            plt.title(f'Split {split_idx+1} Log Loss Curves')
            plt.legend()
            plt.tight_layout()
            plot_dir = plot_base+f"/plots/split{split_idx+1}"
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"loss_error_logloss_curve_epoch_{epoch+1}.png"))
            if epoch:
                prev_plot_path = os.path.join(plot_dir, f"loss_error_logloss_curve_epoch_{epoch+1-20 if epoch+1-20 else 1}.png")
                if os.path.exists(prev_plot_path):
                    os.remove(prev_plot_path)
            # plt.close()
            plt.show()

        wandb.log({
            "epoch": epoch+1,
            "split": split_idx+1,
            "train/epoch_loss": split_train_losses[-1],
            "val/epoch_loss": split_val_losses[-1],
            "train/epoch_abs_rel_error": split_train_abs_rel_errors[-1],
            "val/epoch_abs_rel_error": split_val_abs_rel_errors[-1],
            "train/epoch_log_loss": split_train_log_losses[-1],
            "val/epoch_log_loss": split_val_log_losses[-1]
        })

    split_time = time.time() - split_start_time
    print(f"Split {split_idx+1} completed in {split_time:.2f}s")
    split_dir = model_dir+f"/split{split_idx+1}"
    os.makedirs(split_dir, exist_ok=True)
    torch.save(v_t.state_dict(), os.path.join(split_dir, f"model_split_{split_idx+1}.pth"))
        
# Save the final model
final_model_dir = model_dir+f"/final"
os.makedirs(final_model_dir, exist_ok=True)
torch.save(v_t.state_dict(), os.path.join(final_model_dir, "final_model.pth"))

# Initialize variables for final evaluation
model_mean_mse = None
baseline_mean_mse = None
# Final logging to wandb
num_params = sum(p.numel() for p in v_t.parameters())
trainable_params = sum(p.numel() for p in v_t.parameters() if p.requires_grad)
non_trainable_params = num_params - trainable_params
param_size_bytes = sum(p.element_size() * p.numel() for p in v_t.parameters())
param_size_mb = param_size_bytes / (1024 ** 2)
batch_mem_mb = None  # Initialize batch_mem_mb to None in case it is not computed
total_time = time.time() - global_start_time
print(f"Total execution time: {total_time:.2f} seconds")
wandb.log({
    "test/model_mse": model_mean_mse,
    "test/baseline_mse": baseline_mean_mse,
    "num_params": num_params,
    "trainable_params": trainable_params,
    "non_trainable_params": non_trainable_params,
    "param_size_mb": param_size_mb,
    "batch_mem_mb": batch_mem_mb if 'batch_mem_mb' in locals() else None,
    "total_time": total_time,
})

wandb.save(os.path.join(final_model_dir, "final_model.pth"))
for plot_path in glob.glob(f"scratch/loss_curves/plots/split*/*.png"):
    wandb.log({"loss_curve": wandb.Image(plot_path)
               })
    
print("Training complete. Model saved as 'final_model.pth'.")

# Count the number of model parameters
num_params = sum(p.numel() for p in v_t.parameters())
print(f"Number of model parameters: {num_params}")
# Print model specification details
print("\nModel Specification Details:")
print(f"Model architecture:\n{model}")
print(f"Total number of parameters: {num_params:,}")
trainable_params = sum(p.numel() for p in v_t.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
non_trainable_params = num_params - trainable_params
print(f"Non-trainable parameters: {non_trainable_params:,}")

# Estimate model memory usage (parameters only)
param_size_bytes = sum(p.element_size() * p.numel() for p in v_t.parameters())
param_size_mb = param_size_bytes / (1024 ** 2)
print(f"Estimated memory for model parameters: {param_size_mb:.2f} MB")

# Estimate memory usage for one batch (forward pass)
try:
    # grab one batch from the test loader
    inputs, file_indices, rho_current, _ = next(iter(test_dataloader))
    device = next(v_t.parameters()).device
    example_input = inputs.to(device)

    with torch.no_grad():
        # compute embeddings
        embeddings      = embedding_layer(example_input)      # [B, L, d]
        pos_emb         = pos_embedding_layer(example_input)  # [B, L, d]
        final_embedding = embeddings + pos_emb                # [B, L, d]

        next_file_idx = file_indices[0].item() + 1
        next_file_batch = torch.tensor(flat_dataset.data[torch.randperm
                                    (batch_size,torch.tensor(flat_dataset.file_indices)==
                                                            next_file_idx)], device = device)
        embeddings_tgt = embedding_layer(next_file_batch)
        pos_emb_tgt = pos_embedding_layer(next_file_batch)
        final_embedding_tgt = embeddings_tgt + pos_emb_tgt

        # forward pass
        y_pred = v_t.decode(
            final_embedding_tgt, file_indices
        )  # [m]

    # sum up memory of all intermediate tensors
    tensors = [embeddings, pos_emb, final_embedding,
               next_file_idx, next_file_batch, embeddings_tgt, 
               pos_emb_tgt, final_embedding_tgt, y_pred]
    batch_mem_bytes = sum(t.element_size() * t.nelement() for t in tensors)
    batch_mem_mb = batch_mem_bytes / (1024 ** 2)
    print(f"Estimated memory for one batch (forward pass): {batch_mem_mb:.2f} MB")

except Exception as e:
    print(f"Could not estimate batch memory usage: {e}")
    batch_mem_mb = None

print(f"Device used: {next(v_t.parameters()).device}")

# Evaluate on testset using a metric
v_t.eval()
errors = []
baseline_errors = []
with torch.no_grad():
    for batch, file_indices, rho, rho_next in tqdm(test_dataloader, desc="Testing"):
        filename = flat_dataset.keys[file_indices[0].item()]
        batch = batch.to(device)
        # print(batch[:5])

        embeddings = embedding_layer(batch)
        pos_emb = pos_embedding_layer(batch)
        final_embedding = embeddings + pos_emb

        next_file_idx = file_indices[0].item() + 1
        # Find indices where file_indices == next_file_idx
        candidate_indices = torch.where(torch.tensor(flat_dataset.file_indices) == next_file_idx)[0]
        # print(len(candidate_indices))
        sampled_indices = candidate_indices[torch.randint(0, len(candidate_indices), (batch_size,))]
        next_file_batch = torch.stack(flat_dataset.data).to(device)
        next_file_batch = next_file_batch[sampled_indices]

        # print(next_file_batch[:5])

        embeddings_tgt = embedding_layer(next_file_batch)
        pos_emb_tgt = pos_embedding_layer(next_file_batch)
        final_embedding_tgt = embeddings_tgt + pos_emb_tgt

        file_indices = file_indices.to(device)

        loss = model.loss(v_t, final_embedding, final_embedding_tgt, file_indices)

        if torch.isnan(loss):
            # print("Loss is NaN, skipping this batch.")
            continue
        if torch.isinf(loss):
            # print("Loss is Inf, skipping this batch.")
            continue
        if loss < 0:
            # print("Loss is negative, resetting to zero.")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        if loss > 1000:
            # print("Loss is too high, resetting to 1000.")
            loss = torch.tensor(1000.0, device=device, requires_grad=True)
        y_pred = v_t.decode(final_embedding, file_indices)
        y_true = final_embedding_tgt

        errors.append(loss.cpu())

        # Baseline: output = input (predict log(1) = 0)
        baseline_pred = final_embedding.clone()  # Baseline prediction is the input itself
        baseline_error = torch.mean((baseline_pred - y_true) ** 2)
        baseline_errors.append(baseline_error.cpu())

if errors:
    errors = torch.stack(errors)
    baseline_errors = torch.stack(baseline_errors)
    model_mean_mse = errors.mean().item()
    baseline_mean_mse = baseline_errors.mean().item()
    print(f"Test set Error (model): {model_mean_mse:.4f}")
    print(f"Test set Error (baseline): {baseline_mean_mse:.4f}")
    if model_mean_mse < baseline_mean_mse:
        print("Model outperforms baseline.")
    elif model_mean_mse > baseline_mean_mse:
        print("Baseline outperforms model.")
    else:
        print("Model and baseline perform equally.")
else:
    print("No test samples to evaluate.")
    model_mean_mse = None
    baseline_mean_mse = None

total_time = time.time() - global_start_time
print(f"Total execution time: {total_time:.2f} seconds")

# Final logging to wandb
wandb.log({
    "test/model_mse": model_mean_mse,
    "test/baseline_mse": baseline_mean_mse,
    "num_params": num_params,
    "trainable_params": trainable_params,
    "non_trainable_params": non_trainable_params,
    "param_size_mb": param_size_mb,
    "batch_mem_mb": batch_mem_mb,
    "total_time": total_time,
})

wandb.save(os.path.join(final_model_dir, "final_model.pth"))
for plot_path in glob.glob(f"scratch/loss_curves/plots/split*/*.png"):
    wandb.log({"loss_curve": wandb.Image(plot_path)
               })

wandb.finish()

