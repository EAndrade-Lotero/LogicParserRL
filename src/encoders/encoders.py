import torch

from src.encoders.nn_models import MLP
from src.config.config import DIMS, PATHS

WINDOW_SIZE = DIMS["max_window_size"]
EMBEDDING_DIM = DIMS["embedding_dim"]
ENCODER_DIM = DIMS["encoder_dim"]
HIDDEN_DIMS = DIMS["hidden_dims"]

PATH_SENTENCE_ENCODER = PATHS["sentence_encoder"]
print(f'{PATH_SENTENCE_ENCODER=}')
PATH_DRS_ENCODER = PATHS["drs_encoder"]

class SentenceEncoder():

    def __init__(self):
        pass

    def train(self, data):
        model = MLP(
            input_dim=WINDOW_SIZE,
            hidden_dims=HIDDEN_DIMS,
            out_dim=WINDOW_SIZE,
            dropout=0.2
        )
        x = torch.randn(1, WINDOW_SIZE)
        print(f'{x=}')
        out = model(x)
        print(f'{out=}')
        torch.save(model.state_dict(), PATH_SENTENCE_ENCODER)
        print(f"Model saved to {PATH_SENTENCE_ENCODER}")

    @staticmethod
    def load_from_file(path):
        """
        Loads a SentenceEncoder from a file.
        Args:
            path (str): Path to the file.
        Returns:
            SentenceEncoder: Loaded SentenceEncoder.
        """
        # Load the model from the specified path
        # Instantiate the model
        model = MLP(
            input_dim=WINDOW_SIZE,
            hidden_dims=HIDDEN_DIMS,
            out_dim=WINDOW_SIZE,
            dropout=0.2
        )
        # Load the state_dict
        return model.load_state_dict(torch.load(PATH_SENTENCE_ENCODER))    


class DRSEncoder():

    def __init__(self):
        pass

    def train(self, data):
        model = MLP(
            input_dim=WINDOW_SIZE,
            hidden_dims=HIDDEN_DIMS,
            out_dim=WINDOW_SIZE,
            dropout=0.2
        )
        torch.save(model.state_dict(), PATH_DRS_ENCODER)
        print(f"Model saved to {PATH_DRS_ENCODER}")

    @staticmethod
    def load_from_file(path):
        """
        Loads a SentenceEncoder from a file.
        Args:
            path (str): Path to the file.
        Returns:
            SentenceEncoder: Loaded SentenceEncoder.
        """
        # Load the model from the specified path
        # Instantiate the model
        model = MLP(
            input_dim=WINDOW_SIZE,
            hidden_dims=HIDDEN_DIMS,
            out_dim=WINDOW_SIZE,
            dropout=0.2
        )
        # Load the state_dict
        return model.load_state_dict(torch.load(PATH_DRS_ENCODER))    
