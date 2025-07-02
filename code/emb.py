import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
from torch.utils.data import Sampler
import random
from typing import List, Dict, Optional
import re
from args import get_arg_parser
import argparse
from collections import defaultdict


parser = get_arg_parser()
args = parser.parse_args()
print("Arguments : ",args)

# Use argparse values for all parameters/hyperparameters
dataset_dir = args.dataset_dir
TORCH_SAVE_PATH = args.torch_save_path
USE_TORCH_SAVE = args.use_torch_save
UNIQUE_COUNTS_PATH = args.unique_counts_path
USE_UNIQUE_COUNTS_SAVE = args.use_unique_counts_save
UNIQUE_COUNTS_DIR = args.dataset_dir
embedding_dim = args.embedding_dim
batch_size = args.batch_size
pos_dropout = args.pos_dropout
pos_scale = args.pos_scale
pos_learnable = args.pos_learnable
m = args.m
num_workers = args.num_workers
device = args.device
train_prop = args.train_prop
val_prop = args.val_prop
test_prop = args.test_prop
print(f"Using device: {device}")

# # Check if CUDA is available and set the device accordingly
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

class LookupEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim: int, max_sequence_length: int):
        """
        Lookup embedding layer for nucleotide sequences.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            max_sequence_length (int): Maximum sequence length (L_max in the paper)
        """
        super().__init__()
        
        # Vocabulary mapping as described in the paper
        self.vocab = {
            'CLASS': 0,
            'A': 1,
            'C': 2,
            'G': 3,
            'T': 4,
            'PAD': 5,
            'UNK': 9  # For unknown nucleotides
        }
        self.vocab_size = len(self.vocab)
        
        # Inverse mapping for debugging/interpretability
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        
        # Create embedding matrix (V x d)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.vocab['PAD'],  # PAD token won't contribute to gradient
            device=device  # Ensure embedding is on the correct device
        )
        
    def forward(self, padded_sequences: torch.Tensor) -> torch.Tensor:
        """
        Process batch of already padded sequences through the embedding layer.
        
        Args:
            padded_sequences (torch.Tensor): Tensor of shape (batch_size, L_max + 1)
            
        Returns:
            torch.Tensor: Embedded sequences of shape (batch_size, L_max + 1, d)
        """
        # Get embeddings for all tokens
        embeddings = self.embedding(padded_sequences)  # shape: (batch_size, L_max + 1, d)
        return embeddings
    
    def get_vocabulary(self) -> Dict[str, int]:
        """Return the vocabulary mapping for reference."""
        return self.vocab.copy()

class SequenceDataset(Dataset):
    def __init__(self, directory):
        self.sequences = {}
        self.keys = []
        token_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}  # Matches LookupEmbeddingLayer vocab
        self.invalid_chars = []
        for filename in tqdm(os.listdir(directory)):
            if filename.endswith('.csv'):
                flie_num = filename.split('.')[0]  
                self.sequences[flie_num] = []        
                with open(os.path.join(directory, filename), 'r') as f:
                    lines = f.readlines()[1:]  # skip header
                    for line in lines:
                        seq = line.strip()
                        #Only allow sequences with characters in token_map.keys()
                        # if not all(ch in token_map.keys() for ch in seq):
                        #     print(seq, "contains characters not in token_map. Skipping this sequence.")
                        # Store invalid characters for later reporting
                        self.invalid_chars.extend([ch for ch in seq if ch not in token_map.keys()])
                        if seq and all(ch in token_map.keys() for ch in seq):
                            tokens = [0] + [token_map.get(ch, 9) for ch in seq]  # 0 is CLASS token
                            self.sequences[flie_num].append(torch.tensor(tokens, dtype=torch.long))
        self.sequences = {k: torch.stack(v, 0) for k, v in self.sequences.items() if v}  # Remove empty sequences
        self.keys = list(self.sequences.keys())
        print("Invalid characters found in sequences:", set(self.invalid_chars))

    def __len__(self):
        return len(self.sequences)
    
    def __str__(self):
        result = ""
        for key in self.keys:
            result += "====\n====\n"
            result += f"Dataset : {key}\n\n"
            for seq in self.sequences[key]:
                result += f"{seq.tolist()}\n\n"
            result += "====\n====\n"
        return result
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        return self.sequences[key]

# def custom_collate_fn(batch):
#     # batch is a list of tensors (sequences) from the same file
#     # Pad the sequences to the maximum length in the batch
#     return rnn_utils.pad_sequence(batch, batch_first=True, padding_value=5).to(device)

def custom_collate_fn(batch):
    # batch: list of (sequence_tensor, file_idx)
    sequences, file_indices = zip(*batch)
    # Pad all sequences to max_seq_length using torch.nn.utils.rnn.pad_sequence
    padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=5)
    if padded.size(1) < max_seq_length:
        # If pad_sequence didn't pad to max_seq_length, pad to the right size
        pad_size = max_seq_length - padded.size(1)
        padded = F.pad(padded, (0, pad_size), value=5)
    file_indices = torch.tensor(file_indices, dtype=torch.long)
    return padded.to(device), file_indices

class FlatSequenceDataset(Dataset):
    def __init__(self, sequence_dataset):
        self.data = []
        self.file_indices = []
        self.dict = defaultdict(list)
        self.sorted_keys = sorted(sequence_dataset.keys, key=extract_round_number)
        for file_idx, key in enumerate(self.sorted_keys):
            for seq in sequence_dataset.sequences[key]:
                self.data.append(seq)
                self.file_indices.append(file_idx)
                self.dict[file_idx].append(seq)
        self.keys = self.sorted_keys
        self.dict = dict(self.dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.file_indices[idx]
    
    def __str__(self):
        result = ""
        for idx, (seq, file_idx) in enumerate(zip(self.data, self.file_indices)):
            result += f"Index: {idx}, File Index: {file_idx}, Sequence: {seq.tolist()}\n"
        return result

class SameFileBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # Group indices by file
        self.file_to_indices = {}
        for idx, (_, file_idx) in enumerate(dataset):
            self.file_to_indices.setdefault(file_idx, []).append(idx)
        # Prepare batches
        self.batches = []
        for indices in self.file_to_indices.values():
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]
                self.batches.append(batch)

    def __iter__(self):
        # for batch in self.batches:
        #     yield batch
        yield from self.batches

    def __len__(self):
        return len(self.batches)

class PositionalEmbedding(nn.Module):
    def __init__(self, 
                 embedding_dim: int, 
                 max_sequence_length: int = 1000,
                 dropout: float = 0.1,
                 scale: float = 1.0,
                 learnable: bool = False):
        """
        Sinusoidal positional embedding layer (pure PyTorch implementation).
        
        Args:
            embedding_dim: Dimension of embedding vectors (must be even)
            max_sequence_length: Maximum expected sequence length
            dropout: Dropout probability for embeddings
            scale: Scaling factor for embeddings
            learnable: If True, makes embeddings learnable
        """
        super().__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even"
        
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.scale = scale
        self.learnable = learnable
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize positional encodings
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) * 
            (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        
        pe = torch.zeros(max_sequence_length, embedding_dim)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe * scale
        
        if learnable:
            self.register_parameter('pe', nn.Parameter(pe))
        else:
            self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
               
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Ensure pe is on same device as input
        pe = self.pe.to(x.device)
        
        if x.dim() != 2:
            raise ValueError("Input to PositionalEmbedding must be of shape (batch_size, seq_len)")
        
        batch_size, seq_len = x.shape
        if seq_len > self.max_sequence_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max length {self.max_sequence_length}")
        
        # Expand positional encoding to batch
        pos_emb = pe[:seq_len].unsqueeze(0).expand(batch_size, seq_len, self.embedding_dim)
        return self.dropout(pos_emb)
    
    def get_pe(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Get raw positional embeddings for given length"""
        pe = self.pe
        if device is not None:
            pe = pe.to(device)
        return pe[:seq_len]
    
    def extra_repr(self) -> str:
        return (f"embedding_dim={self.embedding_dim}, max_sequence_length={self.max_sequence_length}, "
                f"learnable={self.learnable}, scale={self.scale}")

def get_unique_sequence_counts(directory):
    """
    Returns a nested dictionary with counts of unique sequences in each file.
    Structure: {filename: {sequence: count, ...}, ...}
    """
    unique_counts = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.csv'):
            file_key = filename.split('.')[0]
            seq_counts = {}
            with open(os.path.join(directory, filename), 'r') as f:
                lines = f.readlines()[1:]  # skip header
                for line in lines:
                    seq = line.strip()
                    if seq:
                        seq_counts[seq] = seq_counts.get(seq, 0) + 1
            unique_counts[file_key] = seq_counts
    return unique_counts

def extract_round_number(key):
    # Extract the number after 'round' in the key
    match = re.search(r'round(\d+)', key)
    return int(match.group(1)) if match else float('inf')


# def get_count_and_abundance(seq_tensor, filename, unique_counts_dict, vocab_inv):
#     seq_tokens = [vocab_inv.get(tok.item(), 'UNK') for tok in seq_tensor if tok.item() != 5]
#     seq_str = ''.join([t for t in seq_tokens[1:] if t in ['A', 'C', 'G', 'T']])
#     file_counts = unique_counts_dict[filename]
#     total_unique = len(file_counts)     # if only unique sequences are to be taken into account
#     # torch.sum(file_counts.values())   # if all the sequences are to be taken into account
#     counts = torch.tensor([file_counts.get(seq, 0) for seq in seq_str], dtype=torch.long)
#     if total_unique > 0:
#         relative_abundance = torch.tensor([round(1e6 * c / total_unique, 4) for c in counts.tolist()])
#     else:
#         relative_abundance = torch.zeros_like(counts, dtype=torch.float)
#     return counts, relative_abundance

def find_batches_with_file_indices(dataloader, target_file_indices):
    """
    Finds and returns the first batch from a dataloader where all file indices are present in the specified target file indices.
    Args:
        dataloader (Iterable): An iterable yielding tuples of (batch, file_indices), where `batch` is a batch of data and `file_indices` is a tensor of file indices corresponding to the batch.
        target_file_indices (torch.Tensor): A tensor containing the file indices to match against.
    Returns:
        list: A list containing a single tuple (batch, file_indices) for the first matching batch where all file indices are in `target_file_indices`. Returns an empty list if no such batch is found.
    Note:
        - Assumes that `device` is defined in the scope where this function is used.
        - Only the first matching batch is returned.
    """
    matching_batches = []
    target_file_indices = target_file_indices.flatten().to(device)  # Ensure on correct device

    for batch, file_indices in dataloader:
        file_indices = file_indices.to(device)
        # Check if all file_indices in batch are in target_file_indices using torch.isin
        mask = torch.isin(file_indices, target_file_indices)
        # print(f"Batch file indices: {file_indices}, Target file indices: {target_file_indices}, Mask: {torch.all(mask)}")
        if torch.all(mask):
            matching_batches.append((batch.to(device), file_indices))
            # print(f"Batch file indices: {file_indices}, Target file indices: {target_file_indices}, Mask: {torch.all(mask)}")
            return matching_batches



# Main execution
# TORCH_SAVE_PATH = 'scratch/dataset.pt'  #'scratch/dataset_trial.pt'
# USE_TORCH_SAVE = True  # Set to True to use torch.save/torch.load

if USE_TORCH_SAVE and os.path.exists(TORCH_SAVE_PATH):
    checkpoint = torch.load(TORCH_SAVE_PATH, weights_only=False)
    dataset = checkpoint['dataset']
    print("Loaded dataset from the file.")
elif USE_TORCH_SAVE:
    dataset = SequenceDataset(dataset_dir)  # 'trial'   'scratch/'+'Selex_csv_set1'
    print(f"Loaded {len(dataset)} sequences.")
    print(f"Loaded sequences from {len(dataset)} file(s).")
    torch.save({'dataset': dataset}, TORCH_SAVE_PATH)
    print("Created and saved new dataset.")

flat_dataset = FlatSequenceDataset(dataset)
# print("File indices for each sequence in flat_dataset:")
# file_indices_tensor = torch.tensor(flat_dataset.file_indices)
# indices_with_1 = (file_indices_tensor == 1).nonzero(as_tuple=True)[0].tolist()
# print("Indices where flat_dataset.file_indices == 1:", indices_with_1[-5:])
# print(flat_dataset.file_indices[282681])
# print(flat_dataset.data[282681])

# batch_size = 1024
batch_sampler = SameFileBatchSampler(flat_dataset, batch_size)

# Create DataLoader with custom collate function
dataloader = DataLoader(flat_dataset, batch_sampler=batch_sampler, collate_fn=custom_collate_fn, 
                        num_workers=num_workers, pin_memory=(device == 'cpu'))
print(f"DataLoader created with {len(dataloader)} batches of size {batch_size}.")
# Note: The custom_collate_fn pads sequences to the maximum length in the batch # Possibly have addressed this in the custom collate function 

# Initialize the embedding layer
# embedding_dim = 8   #128  # Typical embedding dimension
max_seq_length = max([len(seq) for seq in flat_dataset.data])  # Calculate max sequence length
print(f"Max sequence length: {max_seq_length}")
embedding_layer = LookupEmbeddingLayer(embedding_dim=embedding_dim, 
                                     max_sequence_length=max_seq_length).to(device)





# # Usage example:
# target_file_indices = torch.tensor([1, 2, 3])  # Example target tensor
# print("Target file indices:", target_file_indices)
# matching_batches = find_batches_with_file_indices(dataloader, target_file_indices)

# if matching_batches:
#     idx = torch.randint(len(matching_batches), (1,)).item()
#     batch, file_indices = matching_batches[-1]#[idx]
#     print("Randomly selected batch:", batch)
#     print("File indices:", file_indices)
# else:
#     print("No matching batches found.")



# if matching_batches:
#     for batch, file_indices in matching_batches:
#         print("Found batch:", batch)
# else:
#     print("No batches with the specified file_indices.")



   


# UNIQUE_COUNTS_PATH = 'scratch/unique_counts.pt' #'scratch/unique_counts_trial.pt'
# USE_UNIQUE_COUNTS_SAVE = True  # Set to True to use torch.save/torch.load for unique counts
# UNIQUE_COUNTS_DIR = 'scratch/Selex_csv_set1'  # 'trial'

if USE_UNIQUE_COUNTS_SAVE and os.path.exists(UNIQUE_COUNTS_PATH):
    unique_counts = torch.load(UNIQUE_COUNTS_PATH, weights_only=False)
    print("Loaded unique sequence counts from the file.")
elif USE_UNIQUE_COUNTS_SAVE:
    unique_counts = get_unique_sequence_counts(UNIQUE_COUNTS_DIR)
    torch.save(unique_counts, UNIQUE_COUNTS_PATH)
    print("Created and saved unique sequence counts.")
else:
    unique_counts = get_unique_sequence_counts(UNIQUE_COUNTS_DIR)

pos_embedding_layer = PositionalEmbedding(
    embedding_dim=embedding_dim,
    max_sequence_length=max_seq_length,
    dropout=pos_dropout,
    scale=pos_scale,
    learnable=pos_learnable
).to(device)

#m = 1   # 32  # Number of samples per batch to be considered for Query for cross-attention

vocab_inv = {v: k for k, v in embedding_layer.get_vocabulary().items()}

# max_key = max(vocab_inv.keys())  # assume dense keys
# lookup_tensor = torch.full((max_key + 1,), fill_value=-1).cuda()
# for k, v in vocab_inv.items():
#     lookup_tensor[k] = v

def get_counts_and_abundance(batch_tensor, filename, next_filename, unique_counts_dict = unique_counts, vocab_inv = vocab_inv):
    """
    Args:
        batch_tensor: (batch_size, seq_len) tensor of token indices
        filename: str, file key
        unique_counts_dict: dict, unique_counts[filename]
        vocab_inv: dict, index to token mapping

    Returns:
        counts: tensor of shape (batch_size,)
        relative_abundance: tensor of shape (batch_size,)
    """
    seq_strs = []
    for seq_tensor in batch_tensor:
        # Convert tensor to sequence string (skip PAD and CLASS tokens)
        seq_tokens = [vocab_inv.get(tok.item(), 'UNK') for tok in seq_tensor if tok.item() != 5]
        seq_str = ''.join([t for t in seq_tokens[1:] if t in ['A', 'C', 'G', 'T']])
        seq_strs.append(seq_str)

    # seq_strs = lookup_tensor[batch_tensor.cuda()].cpu()

    file_counts = unique_counts_dict[filename]
    next_file_counts = unique_counts_dict[next_filename]
    next_total_unique, total_unique = len(next_file_counts), len(file_counts)
    next_counts, counts = torch.tensor([(next_file_counts.get(seq, 0), file_counts.get(seq, 0)) for seq in seq_strs], dtype=torch.long).T
    if total_unique and next_total_unique:
        relative_abundance, relative_abundance_next = torch.tensor([(round(1e6 * c / total_unique, 4), round(1e6 * c_ / total_unique, 4)) for c,c_ in zip(counts.tolist(), next_counts.tolist())]).T
    elif not total_unique:
        relative_abundance = torch.zeros_like(counts, dtype=torch.float)
    else:
        relative_abundance_next = torch.zeros_like(next_counts, dtype=torch.float)
    return counts, next_counts, relative_abundance, relative_abundance_next

sorted_keys = sorted(flat_dataset.keys, key=extract_round_number)

# # Training loop kinda...for sanity check...
# for batch, file_indices in dataloader:
#     # batch: (batch_size, max_seq_len_in_batch)
#     # file_indices: (batch_size,)
#     batch = batch.to(device)
#     embeddings = embedding_layer(batch)
#     # print(batch)
#     # print(file_indices)    

#     # print(f"Embeddings tensor shape: {embeddings.shape}")
#     vocab_inv = {v: k for k, v in embedding_layer.get_vocabulary().items()}
#     i = torch.randint(0, batch.size(0), (1,)).item()  # Randomly select one sequence in the batch
#     # print(f"Selected sequence index: {i}")
#     # print("Batch shape: ",batch.shape)
#     seq_tensor = batch[i]
#     file_idx = file_indices[i].item()
#     filename = flat_dataset.keys[file_idx]
#     sorted_keys = sorted(flat_dataset.keys, key=extract_round_number)
#     # print(sorted_keys)
#     # print("Not here..!!")
#     # exit(0);
#     # Get counts and abundance for current file as differentiable tensors
#     print(filename, sorted_keys[sorted_keys.index(filename)+1])
#     counts, next_counts, relative_abundance, next_relative_abundance = get_counts_and_abundance(batch, filename, sorted_keys[sorted_keys.index(filename)+1], unique_counts, vocab_inv)
#     # relative_abundance = relative_abundance/torch.sum(relative_abundance)  # Normalize relative abundance
#     count = counts[i]
#     relative_abundance_val = relative_abundance[i]
#     # # Convert tensor to sequence string (skip CLASS token at 0th position)
#     # seq_tokens = [vocab_inv.get(tok.item(), 'UNK') for tok in seq_tensor if tok.item() != 5]
#     # seq_str = ''.join([t for t in seq_tokens[1:] if t in ['A', 'C', 'G', 'T']])

#     # # Get counts and abundance for the next file in sorted_keys (if exists)
#     # try:
#     #     current_idx = sorted_keys.index(filename)
#     #     next_filename = sorted_keys[current_idx + 1]
#     #     next_counts, next_relative_abundance = get_counts_and_abundance(batch, next_filename, unique_counts, vocab_inv)
#     #     next_relative_abundance = next_relative_abundance/torch.sum(next_relative_abundance)  # Normalize relative abundance
#     # except (ValueError, IndexError):
#     #     next_filename = None
#     #     next_counts = torch.zeros_like(counts)
#     #     next_relative_abundance = torch.zeros_like(relative_abundance)
    
#     # For demonstration, get the count and abundance for the selected sequence in the next file
#     next_count = next_counts[i]
#     next_relative_abundance_val = next_relative_abundance[i]

#     pos_emb = pos_embedding_layer(batch)

#     # print(f"Sequence: {seq_str}")
#     # print(f"Filename: {filename}")
#     # print(f"Count in file: {count}")
#     # print(f"No. of unique sequences in file: {len(unique_counts[filename])}")
#     # print(f"Relative abundance in file: {relative_abundance_val} per million")
#     # print(f"Relative abundance in the next file: {next_relative_abundance_val} per million")
#     # print(f"Count in the next file: {next_count}")
#     # print(f"Selected sequence index: {i}")
#     # print(f"Embedding shape: {embeddings.shape}")
#     # print(f"Positional embedding shape: {pos_emb.shape}")
#     # print("Last 5 positions of positional embedding for this sequence:")
#     # print(pos_emb[i, -5:])
#     # print("Example embeddings for the last 5 nucleotides of this sequence:")
#     # print(embeddings[i, -5:])
#     # print("-" * 40)

#     # Combine token embeddings and positional embeddings
#     final_embedding = embeddings + pos_emb
#     # print(f"Final embedding shape: {final_embedding.shape}")
#     # print("Example final embeddings for the last 5 nucleotides of this sequence:")
#     # print(final_embedding[i, -5:])

#     # Only process the first batch for demonstration
#     break

# # Sample m rows from the final_embedding and store them in a tensor
# sample_indices = torch.randperm(final_embedding.size(0))[:m]
# sampled_embeddings = final_embedding[sample_indices]
# # print(f"Sampled {m} embedding(s) shape: {sampled_embeddings.shape}")

# # # Show vocabulary
# # print("\nVocabulary:")
# # for token, idx in embedding_layer.get_vocabulary().items():
#     # print(f"{token}: {idx}")
