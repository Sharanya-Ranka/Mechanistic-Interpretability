## Aim
Use Japanese vs English texts to figure out which of two MI approaches (Sparse Auto Encoders or Linear probing) can achieve a higher performance in clasifying the text.

## Operational details
Use Gemma 2 2B model, and HookedTransformer (TransformerLens) to extract activations

Google GemmaScope - Pretrained SAEs for Gemma models

transformer_lens
- Model names (Gemma 2 2b included) https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.loading_from_pretrained.html#transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES
- 

## Questions to Ponder
- How to collect datapoints? Words within text or average over whole texts? Paired pages only? Same page in Japanese and English.
- Which layer to probe? How many layers does Gemma have? 

## Background Info
SAEs - Encoder-Decoder networks with focus on having "sparse" hidden representation (Disentangle features)