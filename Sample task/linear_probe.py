"""
This module trains a simple Linear Probe (Logistic Regression) to classify
activations as either English or Japanese. This serves as a baseline for
the Sparse Autoencoder analysis.
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import Config
import logging

# Set up logging for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_and_evaluate_linear_probe(activations: torch.Tensor, labels: torch.Tensor):
    """
    Trains a linear probe classifier on the raw activations to classify language.

    Args:
        activations (torch.Tensor): The activations to train the classifier on.
        labels (torch.Tensor): The corresponding labels (0: English, 1: Japanese).
    """
    logger.info("Starting Linear Probe classification.")

    # Reshape activations and labels
    original_shape = activations.shape
    activations_2d = activations.view(-1, original_shape[-1]).cpu().numpy()
    labels_np = labels.repeat_interleave(original_shape[1]).numpy()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        activations_2d, labels_np, test_size=0.2, random_state=42
    )

    # Initialize and train the linear probe
    probe = LogisticRegression(solver="liblinear", max_iter=1000)
    logger.info("Training the Logistic Regression probe...")
    probe.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = probe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("--- Linear Probe Results ---")
    logger.info(f"Classification accuracy on test set: {accuracy * 100:.2f}%")
    logger.info(
        "This score serves as a baseline for language classification on raw activations."
    )
