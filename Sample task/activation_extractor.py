"""
This module handles the core task of loading a Gemma 2 2B model using the
TransformerLens library, tokenizing text, and extracting activations from a
specified transformer layer. The activations are then saved to a file.
"""

import os
import torch
import logging
from transformer_lens import HookedTransformer
from wiki_data_fetcher import fetch_wiki_articles
from config import Config
from tqdm import tqdm

# Set up logging for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_activations_from_text(texts, model, layer_index):
    """
    Passes text through the model and extracts the activations of a specific layer.

    Args:
        texts (list): A list of text strings.
        model (HookedTransformer): The loaded TransformerLens model.
        layer_index (int): The index of the layer to extract activations from.

    Returns:
        torch.Tensor: A tensor of activations for all texts.
    """
    all_activations = []
    logger.info(f"Extracting activations from layer {layer_index}...")

    with tqdm(total=len(texts), desc="Extracting Activations") as pbar:
        for text in texts:
            # We use `run_with_cache` to get the activations
            # The cache contains all the intermediate activations of the model.
            logits, cache = model.run_with_cache(text, prepend_bos=True)
            # Get the residual stream activations from the specified layer
            activations = cache[f"blocks.{layer_index}.hook_resid_post"]
            all_activations.append(activations.detach())
            pbar.update(1)

    # Concatenate activations from all texts
    # The shape will be (batch_size, sequence_length, activation_dim)
    return torch.cat(all_activations, dim=0)


def extract_and_save_activations():
    """
    Main function to fetch data, extract activations, and save them.
    """
    logger.info("Starting activation extraction process...")

    # 1. Fetch data from Wikipedia
    wiki_data = fetch_wiki_articles(
        Config.EN_ARTICLE_TITLE, Config.JA_ARTICLE_TITLE, Config.NUM_PARAGRAPHS
    )
    if not wiki_data:
        logger.error("Failed to fetch Wikipedia data. Exiting.")
        return None

    # Prepare texts and labels
    en_text_lines = [
        line.strip() for line in wiki_data["english"].split("\n") if line.strip()
    ]
    ja_text_lines = [
        line.strip() for line in wiki_data["japanese"].split("\n") if line.strip()
    ]
    texts = en_text_lines + ja_text_lines
    labels = [0] * len(en_text_lines) + [1] * len(
        ja_text_lines
    )  # 0 for English, 1 for Japanese

    # 2. Load the model using TransformerLens
    logger.info(f"Loading model: {Config.MODEL_NAME} on device: {Config.DEVICE}")
    breakpoint()
    try:
        model = HookedTransformer.from_pretrained(
            Config.MODEL_NAME, device=Config.DEVICE, center_unembed=False
        )
    except Exception as e:
        logger.error(f"Failed to load model. Please check your setup. Error: {e}")
        return None

    # 3. Get activations from the model
    raw_activations = get_activations_from_text(texts, model, Config.TARGET_LAYER_INDEX)

    # 4. Save the activations and labels
    data_to_save = {
        "activations": raw_activations.cpu(),
        "labels": torch.tensor(labels),
        "metadata": {
            "model_name": Config.MODEL_NAME,
            "layer_index": Config.TARGET_LAYER_INDEX,
            "en_article": Config.EN_ARTICLE_TITLE,
            "ja_article": Config.JA_ARTICLE_TITLE,
        },
    }
    torch.save(data_to_save, Config.ACTIVATION_FILE_PATH)
    logger.info(f"Activations saved to {Config.ACTIVATION_FILE_PATH}")
    return Config.ACTIVATION_FILE_PATH


def load_activations():
    """
    Loads activations from the saved file.
    """
    if not os.path.exists(Config.ACTIVATION_FILE_PATH):
        logger.error(f"Activations file not found at {Config.ACTIVATION_FILE_PATH}")
        return None
    try:
        data = torch.load(Config.ACTIVATION_FILE_PATH)
        logger.info(f"Activations loaded from {Config.ACTIVATION_FILE_PATH}")
        return data["activations"], data["labels"]
    except Exception as e:
        logger.error(f"Failed to load activations file. Error: {e}")
        return None, None
