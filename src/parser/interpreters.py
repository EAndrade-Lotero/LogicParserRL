import json
import numpy as np

from typing import List
from pathlib import Path
from numpy import (
    array as np_array,
    concatenate as np_concatenate,
    float32 as np_float32
)

from parser.parser_auxiliary_classes import Nodo
from config.config import encoder_model


def id_state(state):
    '''
    Default interpreter: do nothing.
    '''
    return state

def parser_interpreter(state):
    '''Create an embedding for the state using the encoder model.'''
    estado_actual, raiz = state['Estado'], state['Raiz'] 
    frase = estado_actual.frase
    indice = estado_actual.indice
    nivel = estado_actual.nodo.nivel()
    cadena = estado_actual.obtener_cadena()
    msg = f"{indice}\n{nivel}\n{frase}"
    try:
        formula = str(raiz.simplificar2recompensa().fol())
    except Exception as e:
        print(f"Error simplifying Nodo: {raiz}")
        raise e
    # msg = f"{frase}\n{indice}\n{nivel}\n{cadena}\n{formula}"
    embeddings = encoder_model.encode(
        [
            cadena,
            frase,
            formula
        ],
        convert_to_numpy=True
    ).flatten()
    indice = np_array([estado_actual.indice]).astype(np_float32)
    nivel = np_array([estado_actual.nodo.nivel()]).astype(np_float32)
    embeddings = np_concatenate([embeddings, indice, nivel])
    return embeddings

class dict_embedding_interpreter:

    file_path = Path('embeding_dict.json')

    def __init__(self) -> None:
        with open(self.file_path, 'r') as f:
            self.dict_embeddings = json.load(f)
        
    def __call__(self, list_str:List[str]) -> np.ndarray:
        embeddings = [
            self.dict_embeddings.get(sentence, None)
                for sentence in list_str 
        ]
        unknown_sentences = [
            list_str[idx] for idx, x in enumerate(embeddings)
                if x is None
        ]
        if len(unknown_sentences) > 0:
            new_embeddings = encoder_model.encode(
                unknown_sentences,
                convert_to_numpy=True
            )
            self.dict_embeddings.update(dict(zip(unknown_sentences, new_embeddings))
        )