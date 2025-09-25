"""
This is the main runner file that orchestrates the entire program.
It imports functions from other modules and executes them in sequence.
"""

import logging
import os
from activation_extractor import extract_and_save_activations, load_activations
from linear_probe import train_and_evaluate_linear_probe
from sae import train_sae_and_analyze
from config import Config

# Configure basic logging for the main runner
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_task():
    """
    Executes the entire workflow:
    1. Fetches data and extracts activations if they don't exist.
    2. Trains and evaluates a linear probe.
    3. Trains and analyzes a Sparse Autoencoder.
    """
    # --- Step 1: Extract Text and Activations ---
    # Check if the activation file already exists. If so, skip this step.
    if os.path.exists(Config.ACTIVATION_FILE_PATH):
        logger.info(
            f"Found existing activation file at {Config.ACTIVATION_FILE_PATH}. Skipping extraction."
        )
        activations, labels = load_activations()
    else:
        logger.info("No activation file found. Starting extraction process.")
        # breakpoint()
        file_path = extract_and_save_activations()
        if not file_path:
            logger.error("Activation extraction failed. Exiting.")
            return
        activations, labels = load_activations()

    if activations is None or labels is None:
        logger.error("Failed to load activations for subsequent tasks. Exiting.")
        return

    # --- Step 2: Classify with Linear Probe (Baseline) ---
    logger.info("\nRunning Linear Probe on raw activations...")
    train_and_evaluate_linear_probe(activations, labels)

    # --- Step 3: Train and Analyze Sparse Autoencoder ---
    logger.info("\nRunning Sparse Autoencoder training and analysis...")
    train_sae_and_analyze(activations, labels)


if __name__ == "__main__":
    run_task()
