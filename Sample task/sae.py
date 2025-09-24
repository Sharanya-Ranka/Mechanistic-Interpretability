"""
This module implements a Sparse Autoencoder (SAE) for activations and
analyzes its hidden activations to find a dimension encoding the "IsJapanese" feature.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import Config
from tqdm import tqdm
import logging

# Set up logging for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SparseAutoencoder(nn.Module):
    """
    A simple Sparse Autoencoder model.
    """

    def __init__(self, activation_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(activation_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, activation_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        # Use ReLU as the non-linearity for sparsity
        sparse_hidden = self.relu(encoded)
        reconstructed = self.decoder(sparse_hidden)
        return reconstructed, sparse_hidden


def train_sae_and_analyze(activations: torch.Tensor, labels: torch.Tensor):
    """
    Trains a Sparse Autoencoder on activations and then analyzes the
    hidden dimensions.

    Args:
        activations (torch.Tensor): The activations to train the SAE on.
        labels (torch.Tensor): The corresponding labels (0: English, 1: Japanese).
    """
    logger.info("Starting Sparse Autoencoder training and analysis.")

    # Reshape activations to be a 2D tensor (samples, features)
    # The activations are likely (batch_size, seq_len, activation_dim), we need to flatten
    original_shape = activations.shape
    activations_2d = activations.view(-1, original_shape[-1]).to(Config.DEVICE)

    # Normalize activations to have unit variance
    activations_2d = activations_2d / activations_2d.std()

    # Create dataset and dataloader
    dataset = TensorDataset(activations_2d)
    dataloader = DataLoader(dataset, batch_size=Config.SAE_BATCH_SIZE, shuffle=True)

    # Initialize the SAE and optimizer
    sae = SparseAutoencoder(Config.ACTIVATION_DIM, Config.SAE_HIDDEN_DIM).to(
        Config.DEVICE
    )
    optimizer = torch.optim.Adam(sae.parameters(), lr=Config.SAE_LEARNING_RATE)
    mse_loss = nn.MSELoss()

    # Training loop
    logger.info(f"Training SAE for {Config.SAE_EPOCHS} epochs...")
    for epoch in tqdm(range(Config.SAE_EPOCHS), desc="SAE Training"):
        total_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            reconstructed, sparse_hidden = sae(inputs)

            # Reconstruction Loss (MSE)
            reconstruction_loss = mse_loss(reconstructed, inputs)

            # Sparsity Loss (L1 regularization on the hidden activations)
            sparsity_loss = torch.norm(sparse_hidden, p=1, dim=1).mean()

            # Total Loss
            loss = reconstruction_loss + Config.SAE_SPARSITY_LAMBDA * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    logger.info(
        f"SAE training complete. Final loss: {total_loss / len(dataloader):.4f}"
    )

    # Analysis: Find the hidden dimension that best encodes "IsJapanese"
    logger.info("Analyzing SAE hidden dimensions for 'IsJapanese' feature.")

    with torch.no_grad():
        _, sae_hidden_activations = sae(activations_2d)

    # Detach from GPU and convert to numpy for scikit-learn
    sae_hidden_activations_np = sae_hidden_activations.cpu().numpy()
    labels_np = labels.repeat_interleave(original_shape[1]).numpy()

    # We will use a linear probe (Logistic Regression) for each hidden dimension
    # and check its accuracy. The best one is likely to be the "IsJapanese" feature.
    best_accuracy = -1
    best_dim_index = -1

    for i in tqdm(range(Config.SAE_HIDDEN_DIM), desc="Probing hidden dimensions"):
        # Get the activations for a single hidden dimension
        single_dim_activations = sae_hidden_activations_np[:, i].reshape(-1, 1)

        # Train a logistic regression probe
        probe = LogisticRegression(max_iter=1000)
        # Handle cases where the activations for a dimension are all the same
        if len(np.unique(single_dim_activations)) > 1:
            probe.fit(single_dim_activations, labels_np)
            accuracy = probe.score(single_dim_activations, labels_np)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_dim_index = i

    logger.info("--- SAE Analysis Results ---")
    if best_dim_index != -1:
        logger.info(
            f"The best hidden dimension for 'IsJapanese' is dimension {best_dim_index}"
        )
        logger.info(
            f"This dimension's activations can classify English/Japanese with an accuracy of {best_accuracy * 100:.2f}%"
        )
    else:
        logger.info("No single SAE dimension could effectively classify the language.")
