"""
Configuration file for the activation extraction and analysis program.
This module stores all the global variables and settings needed for the tasks.
"""

import torch


class Config:
    """
    A class to hold all configuration settings for the program.
    """

    # --- Data Fetching Settings ---
    # Wikipedia article titles to fetch in English and Japanese
    EN_ARTICLE_TITLE = "Artificial intelligence"
    JA_ARTICLE_TITLE = "人工知能"  # Japanese for "Artificial intelligence"
    # Number of paragraphs to extract from each article
    NUM_PARAGRAPHS = 10
    WIKI_DUMP_FILEPATH = "wiki_data_dump.json"

    # --- Model and Activation Extraction Settings ---
    # The Hugging Face model to load for activation extraction.
    # We use a Gemma 2B model for this task.
    MODEL_NAME = "google/gemma-2-2b"
    LOCAL_MODEL_FILEPATH = "/content/drive/MyDrive/Gemma2_2bModel"
    # The index of the transformer layer from which to extract activations.
    # Note: Gemma 2B has 18 layers (0 to 17).
    TARGET_LAYER_INDEX = 17

    # --- File Paths ---
    # Path to save the extracted activations.
    ACTIVATION_FILE_PATH = "gemma2_activations_l{}.pt".format(TARGET_LAYER_INDEX)

    # --- Sparse Autoencoder Settings ---
    # Dimensionality of the original activations
    ACTIVATION_DIM = 2048
    # Dimensionality of the sparse hidden layer (should be much larger than ACTIVATION_DIM)
    SAE_HIDDEN_DIM = 8192
    # The sparsity penalty coefficient (lambda) for the L1 loss
    SAE_SPARSITY_LAMBDA = 0.001
    # Number of training epochs for the SAE
    SAE_EPOCHS = 50
    # Batch size for SAE training
    SAE_BATCH_SIZE = 64
    # Learning rate for the SAE
    SAE_LEARNING_RATE = 1e-4

    # --- Device Configuration ---
    # Use GPU if available, otherwise fallback to CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Logging and Verbosity ---
    VERBOSITY_LEVEL = 1  # 0: no logs, 1: basic logs, 2: detailed logs
