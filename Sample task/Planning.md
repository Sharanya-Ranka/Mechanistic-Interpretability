## Aim
Use Japanese vs English texts to figure out which of two MI approaches (Sparse Auto Encoders or Linear probing) can achieve a higher performance in finding the feature.

## Operational details
Use Gemms 2B model
Use Google Colab
HookedTransformer from the transformer_lens module to extract midway activations
SAE stuff too

transformer_lens
- Model names (Gemma 2 2b included) https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.loading_from_pretrained.html#transformer_lens.loading_from_pretrained.OFFICIAL_MODEL_NAMES
- 

## Questions to Ponder
- How to collect datapoints? Words within text or average over whole texts? Paired pages only? Same page in Japanese and English.
- Which layer to probe? How many layers does Gemma have? 

## Background Info
SAEs - Encoder-Decoder networks with focus on having "sparse" hidden representation (Disentangle features)