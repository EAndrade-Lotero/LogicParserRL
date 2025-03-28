from pathlib import Path

DIMS = {
    "max_window_size": 10,
    "embedding_dim": 16,
    "encoder_dim": 16,
    "hidden_dims": [8, 8]
}

# Path to the current file
src_dir = Path(__file__).parent / Path("..")
src_dir = src_dir.resolve()
print(f'{src_dir=}')
new_path = Path(src_dir) / Path("/data/encoder_data")
print(f'{new_path=}')

PATHS = {
    "sentence_encoder": Path("../src/data/encoder_data", "sentence_encoder.pth"),
    "drs_encoder": src_dir / Path("../src/data/encoder_data/drs_encoder.pth"),
}
