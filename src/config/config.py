from pathlib import Path
from torch.cuda import is_available as cuda_available
from sentence_transformers import SentenceTransformer


# Embedding model for the parser
# This model is used to encode observarion space (state) of the parser.
model_name = 'multi-qa-MiniLM-L6-dot-v1'
# model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
device = 'cuda' if cuda_available() else 'cpu'

encoder_model = SentenceTransformer(model_name, device=device)
tokenizer = encoder_model.tokenizer

# Special tokens
ENCODER_DIM = encoder_model.get_sentence_embedding_dimension()
EMB_DIM = ENCODER_DIM * 3 + 2  # +2 for the index of the state and the level of the node
# EMB_DIM = ENCODER_DIM * 4
SEP = tokenizer.sep_token
MASK = tokenizer.mask_token
PAD = tokenizer.pad_token
UNK = tokenizer.unk_token

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
