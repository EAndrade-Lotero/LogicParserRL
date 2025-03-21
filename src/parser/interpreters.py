from sentence_transformers import SentenceTransformer

encoder_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
tokenizer = encoder_model.tokenizer
SEP = tokenizer.sep_token

def id_state(state):
    '''
    Default interpreter: do nothing.
    '''
    return state

def parser_interpreter(state):
    estado_actual, raiz = state['Estado'], state['Raiz'] 
    state_str = estado_actual.obtener_cadena() + f"\n {raiz}"
    embeddings = encoder_model.encode(
        [estado_actual.frase, SEP + estado_actual.obtener_cadena() + SEP, str(raiz)],
        convert_to_numpy=True
    ).flatten()
    return embeddings