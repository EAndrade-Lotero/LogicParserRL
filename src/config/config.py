from pathlib import Path

DIMS = {
    "max_window_size": 10,
    "embedding_dim": 8,
    "encoder_dim": 8,
    "hidden_dims": [8, 8]
}

# Path to the current file
src_dir = Path(__file__).parent / Path("..")
src_dir = src_dir.resolve()

PATHS = {
    "sentence_encoder": Path(src_dir, "../src/data/encoder_data", "sentence_encoder.pth").resolve(),
    "drs_encoder": Path(src_dir, "../src/data/encoder_data/drs_encoder.pth").resolve(),
    "training_data_folder": Path(src_dir, "../src/data/training_data/").resolve(),
    "tokenizer_folder": Path(src_dir, "../src/data/tokenizers/").resolve(),
    "grammar_folder": Path(src_dir, "../src/data/grammars/").resolve(),
}
