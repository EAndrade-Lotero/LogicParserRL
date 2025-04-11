import torch
from tqdm.auto import tqdm

from encoders.nn_models import MLP
from config.config import DIMS, PATHS

WINDOW_SIZE = DIMS["max_window_size"]
EMBEDDING_DIM = DIMS["embedding_dim"]
ENCODER_DIM = DIMS["encoder_dim"]
HIDDEN_DIMS = DIMS["hidden_dims"]

PATH_SENTENCE_ENCODER = PATHS["sentence_encoder"]
PATH_DRS_ENCODER = PATHS["drs_encoder"]

class SentenceEncoder():

    def __init__(self):
        pass

    def train(self, dataloader, device):
        """
        Train the model for the specified number of epochs.
        
        Args:
            dataloader: DataLoader providing data for training.

        Returns:
            model: The trained model.
            epoch_losses: List of average losses for each epoch.
        """
        model = MLP(
            input_dim=WINDOW_SIZE,
            hidden_dims=HIDDEN_DIMS,
            out_dim=WINDOW_SIZE,
            dropout=0.2
        )

        model.to(device)  # Move model to the specified device

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    
        num_epochs = 10
        # List to store running loss for each epoch
        epoch_losses = []

        for epoch in tqdm(range(num_epochs)):
            # Storing running loss values for the current epoch
            running_loss = 0.0

            # Using tqdm for a progress bar
            for input, target in dataloader:
                optimizer.zero_grad()               
                predicted = model(input)                               
                loss = criterion(predicted, target)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                running_loss += loss.item()

            # Append average loss for the epoch
            epoch_losses.append(running_loss / len(dataloader))
        
        return model, epoch_losses
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
