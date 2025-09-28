"""
This module handles the core task of loading a Gemma 2 2B model using the
TransformerLens library, tokenizing text, and extracting activations from a
specified transformer layer. The activations are then saved to a file.
"""

import os
import torch
import logging

# from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from wiki_data_fetcher import fetch_wiki_articles
from config import Config
import json
from tqdm import tqdm

# Set up logging for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_activations_from_text(text, model, tokenizer, layer_index):
    """
    Passes text through the model and extracts the activations of a specific layer.

    Args:
        texts (str): A string
        model (HookedTransformer): The loaded TransformerLens model.
        layer_index (int): The index of the layer to extract activations from.

    Returns:
        torch.Tensor: A tensor of activations for all texts.
    """
    # all_activations = []
    # logger.info(f"Extracting activations from layer {layer_index}...")

    # with tqdm(total=len(texts), desc="Extracting Activations") as pbar:
    #     for text in texts:
    #         # We use `run_with_cache` to get the activations
    #         # The cache contains all the intermediate activations of the model.
    #         logits, cache = model.run_with_cache(text, prepend_bos=True)
    #         # Get the residual stream activations from the specified layer
    #         activations = cache[f"blocks.{layer_index}.hook_resid_post"]
    #         all_activations.append(activations.detach())
    #         pbar.update(1)

    # # Concatenate activations from all texts
    # # The shape will be (batch_size, sequence_length, activation_dim)
    # return torch.cat(all_activations, dim=0)

    def gather_residual_activations(model, inputs):
        target_act = []
        all_handles = []

        def gather_target_act_hook(mod, inputs, outputs):
            nonlocal target_act  # make sure we can modify the target_act from the outer scope
            target_act.append(outputs[0])
            return outputs

        for layer_ind in range(26):
            all_handles.append(
                model.model.layers[layer_ind].register_forward_hook(
                    gather_target_act_hook
                )
            )

        _ = model.forward(inputs)

        for handle in all_handles:
            handle.remove()

        return torch.cat(target_act, dim=0)

    inputs = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True).to(
        "cuda"
    )

    # breakpoint()
    target_act = gather_residual_activations(model, inputs)

    return target_act


def extract_and_save_activations():
    """
    Main function to fetch data, extract activations, and save them.
    """
    logger.info("Starting activation extraction process...")

    torch.set_grad_enabled(False)

    # 1. Fetch data from Wikipedia
    wiki_dump_filepath = Config.WIKI_DUMP_FILEPATH
    with open(wiki_dump_filepath, "r", encoding="utf-8") as f:
        # Use ensure_ascii=False to correctly handle Japanese characters
        wiki_data = json.load(f)
        # json.dump(data, f, ensure_ascii=False, indent=4)

    if not wiki_data:
        logger.error("Failed to fetch Wikipedia data. Exiting.")
        return None

    # Prepare texts and labels
    en_text_lines = [
        line.strip() for line in wiki_data["english"].split("\n") if line.strip()
    ][:2]
    ja_text_lines = [
        line.strip() for line in wiki_data["japanese"].split("\n") if line.strip()
    ][:2]
    texts = en_text_lines + ja_text_lines
    labels = [0] * len(en_text_lines) + [1] * len(
        ja_text_lines
    )  # 0 for English, 1 for Japanese

    # breakpoint()
    # 2. Load the model using TransformerLens
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    local_model_filepath = Config.LOCAL_MODEL_FILEPATH

    tokenizer = AutoTokenizer.from_pretrained(local_model_filepath)
    # model = None
    model = AutoModelForCausalLM.from_pretrained(
        local_model_filepath,
        device_map="auto",
        # quantization_config=quantization_config,
    )

    # Check how many layers in the model
    # Can you enable quantization using Prud's config?
    # breakpoint()

    # 3. Get activations from the model
    en_text_activations = []
    for en_ind, en_text in enumerate(en_text_lines):
        print(f"English {en_ind}")
        raw_activations = get_activations_from_text(
            en_text, model, tokenizer, Config.TARGET_LAYER_INDEX
        )
        en_text_activations.append(raw_activations.cpu())

    ja_text_activations = []
    for ja_ind, ja_text in enumerate(ja_text_lines):
        print(f"Japanese {ja_ind}")
        raw_activations = get_activations_from_text(
            ja_text, model, tokenizer, Config.TARGET_LAYER_INDEX
        )
        ja_text_activations.append(raw_activations.cpu())

    breakpoint()

    # 4. Save the activations and labels
    data_to_save = {
        "en_text_activations": en_text_activations,
        "ja_text_activations": ja_text_activations,
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
