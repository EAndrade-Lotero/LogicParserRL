import torch
from sentence_transformers import SentenceTransformer

from src.encoders.encoders import SentenceEncoder, DRSEncoder
from src.config.config import PATHS

model_name = 'multi-qa-MiniLM-L6-dot-v1'
# model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_model = SentenceTransformer(model_name, device=device)
tokenizer = encoder_model.tokenizer
SEP = tokenizer.sep_token

PATH_SENTENCE_ENCODER = PATHS["sentence_encoder"]
PATH_DRS_ENCODER = PATHS["drs_encoder"]

class ParserInterpreter:
    '''
    Parser interpreter interface.
    '''
    def __init__(self):
        self.sentence_encoder = SentenceEncoder.load_from_file(PATH_SENTENCE_ENCODER)
        self.drs_encoder = DRSEncoder.load_from_file(PATH_DRS_ENCODER)

    def get_embedding(self, estado) -> torch.Tensor:
        """
        Obtiene el embedding de la DRS actual.
        Returns:
            np.ndarray: Embedding de la DRS actual.
        """
        # Inicializa el embedding con ceros
        nodo_indice = self.estado.get_nodo_indice()
        # Crea el embedding de los índices
        embed_indices = torch.tensor(
            [
                estado.indice # indice de la palabra actual
                nodo_indice # indice del nodo actual
            ],
            dtype=torch.float32
        )
        # Crea el embedding de la frase y el DRS
        frase = estado.frases[nodo_indice]
        embedding_sentence = self.sentence_encoder(frase)
        embedding_drs = self.drs_encoder(estado.get_nodo().drs.simplify())
        # Concatena los embeddings
        embedding = torch.stack((embed_indices, embedding_sentence, embedding_drs))
        return embedding



