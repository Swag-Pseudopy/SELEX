import argparse
import torch
import os
import sys


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Sequence Embedding Pipeline Arguments")
    parser.add_argument('--dataset_dir', type=str, default='scratch/Selex_csv_set1',  # 'trial'
                        help='Directory containing the sequence CSV files')
    parser.add_argument('--torch_save_path', type=str, default='scratch/dataset.pt',  # 'dataset_trial.pt'
                        help='Path to save/load the torch dataset')
    parser.add_argument('--use_torch_save', action='store_true', default=True,
                        help='Whether to use torch.save/torch.load for dataset')
    parser.add_argument('--unique_counts_path', type=str, default='scratch/unique_counts.pt', # 'scratch/unique_counts_trial.pt'
                        help='Path to save/load the unique counts dictionary')
    parser.add_argument('--use_unique_counts_save', action='store_true', default=True,
                        help='Whether to use torch.save/torch.load for unique counts')
    parser.add_argument('--embedding_dim', type=int, default=128, # 8
                        help='Dimension of the embedding vectors')
    parser.add_argument('--batch_size', type=int, default=1024, # 256, # 512,
                        metavar='N',
                        help='Batch size for DataLoader')
    parser.add_argument('--pos_dropout', type=float, default=0.0,
                        help='Dropout probability for positional embedding')
    parser.add_argument('--pos_scale', type=float, default=1.0,
                        help='Scaling factor for positional embedding')
    parser.add_argument('--pos_learnable', action='store_true', default=True,
                        help='Whether positional embedding is learnable')
    parser.add_argument('--m', type=int, default=128, # 1
                        metavar='m',
                        help='Number of samples per batch to be considered for Query for cross-attention')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device to use for computation (e.g., "cpu", "cuda:0")')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads for the cross-attention model')
    parser.add_argument('--num_workers', type=int, default=0,   # 4
                        help='Number of worker threads for DataLoader')
    parser.add_argument('--num_splits', type=int, default=8,
                        help='Number of Splits to be used for the dataset')
    parser.add_argument('--num_epochs', type=int, default=100, # 10
                        help='Number of epochs for training the model')
    parser.add_argument('--train_prop', type=float, default=0.8,
                        help='Proportion of the dataset to use for training')
    parser.add_argument('--val_prop', type=float, default=0.1,
                        help='Proportion of the dataset to use for validation')
    parser.add_argument('--test_prop', type=float, default=0.1,
                        help='Proportion of the dataset to use for testing')
    parser.add_argument('--num_layers', type=int, default=1, #4
                        help='Number of layers in the model')
    parser.add_argument('--num_cross_att', type=int, default=1, #2
                        help='Number of cross-attention layers in the model')
    parser.add_argument('--plot_base', type=str, default='scratch/plots_flow3', #'plots_MSE', #'plots_KL_norm'#'plots_RCA'
                        help='Path to save/load the torch dataset')
    parser.add_argument('--model_dir', type=str, default='scratch/model_flow3',  #'model_MSE', #'model_KL_norm', #'model_RCA'
                        help='Path to save/load the torch dataset')
    parser.add_argument('--loss', type=str, choices=['MSE', 'KL'], default='MSE',
                        help='Loss function to use: "MSE" for Mean Squared Error or "KL" for Kullback-Leibler divergence (default: KL)')
    parser.add_argument('--round_embedding_dim', type=int, default=16,
                        help='Round number for embedding dimension')
    parser.add_argument('--n_freqs', type=int, default=8,
                        help='Number of frequency bands for positional encoding')
    parser.add_argument('--num_tren', type=int, default=1,
                        help='Number of layers of transformer encoder')
    parser.add_argument('--mid_layer_abundance_injection', action='store_true', default=True,
                        help='Whether to inject abundance information at the mid layer')
    return parser
    
