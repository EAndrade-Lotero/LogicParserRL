import stanza
import numpy as np
import nltk
import nltk.sem.drt as drt

from copy import deepcopy
from typing import List, Dict, Tuple, Optional
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete

dexpr = drt.DrtExpression.fromstring
nlp = stanza.Pipeline(
    lang='es', 
    processors='tokenize,pos,lemma', 
    verbose=False
)


class Nodo:
    """Representa el nodo de un arbol 3-aridad para la construcción de DRS.
    Atributos:
        drs (drt.DrtExpression): Representacion inicial del nodo en forma de DRS
        antecedente (Nodo | None): Nodo hijo que representa el antecedente de la DRS
        consecuente (Nodo | None): Nodo hijo que representa el consecuente de la DRS
        negacion (Nodo | None): Nodo hijo que representa la negación de la DRS
        madre (Nodo | None): Nodo madre del nodo actual
    """
    def __init__(self):
        """
        Inicializa un nodo con una DRS vacía y sin hijos.
        """
        self.drs = dexpr('([],[])')
        self.antecedente = None
        self.consecuente = None
        self.negacion = None
        self.madre = None
        
    def __str__(self):
        """
        Retorna la representación simplificada en cadena de la DRS del nodo.
        Returns:
            str: Representación simplificada de la DRS del nodo.
        """
        return str(self.simplificar())
        
    @staticmethod
    def crear_negacion(madre:'Nodo') -> 'Nodo':
        """
        Crea un nodo hijo que representa la negación de la DRS del nodo madre.
        Args:
            madre (Nodo): Nodo madre del nodo a crear.
        Returns:
            Nodo: Nuevo nodo hijo de negación.
        """
        nodo = Nodo()
        nodo.madre = madre
        madre.negacion = nodo
        return nodo
    
    @staticmethod
    def crear_antecedente(madre:'Nodo') -> 'Nodo':
        """
        Crea un nodo hijo que representa el antecedente enlazado al nodo madre.
        Args:
            madre (Nodo): Nodo madre del nodo a crear.
        Returns:
            Nodo: Nuevo nodo hijo de antecedente.
        """
        nodo = Nodo()
        nodo.madre = madre
        madre.antecedente = nodo
        return nodo
    
    @staticmethod
    def crear_consecuente(madre:'Nodo') -> 'Nodo':
        """
        Crear un nodo hijo que representa el consecuente enlazado al nodo madre.
        Args:
            madre (Nodo): Nodo madre del nodo a crear.
        Returns:
            Nodo: Nuevo nodo hijo de consecuente.
        """
        nodo = Nodo()
        nodo.madre = madre
        madre.consecuente = nodo
        return nodo
    
    def simplificar(self) -> drt.DrtExpression:
        """
        Simplifica la DRS del nodo considerando sus conexiones logicas (hijos)
        Returns:
            drt.DrtExpression: Expresión DRS simplificada del nodo.
        """
        drs = deepcopy(self.drs)
        if self.negacion is not None:
            drs_interna_neg = str(self.negacion.simplificar())
            drs += dexpr(f'([][-{drs_interna_neg}])')
        if self.antecedente is not None and self.consecuente is not None:
            drs_antecedente = str(self.antecedente.simplificar())
            drs_consecuente = str(self.consecuente.simplificar())
            drs += dexpr(f'([][({drs_antecedente} -> {drs_consecuente})])')
        return drs.simplify()
    
    def simplificar2recompensa(self) -> drt.DrtExpression:
        """
        Variante de la simplificación que adapta la estructura DRS a la formulación de recompensa.
        En esta simplificacion se establece una estructura de representación 
        de la DRS que considera las conexiones lógicas no completas de los 
        nodos hijos. Es decir, se establece un tratamiento especifico para los 
        casos en los que los nodos hijos antecedente y consecuente no estan completos.

        Se establece esta estructura para la comparación de la DRS actual con la DRS objetivo.

        Returns:
            drt.DrtExpression: Expresión DRS simplificada del nodo.
        """
        drs = deepcopy(self.drs)
        if self.negacion is not None:
            drs_interna_neg = str(self.negacion.simplificar2recompensa())
            drs += dexpr(f'([][-{drs_interna_neg}])')
        if self.antecedente is not None and self.consecuente is not None:
            drs_antecedente = self.antecedente.simplificar2recompensa()
            drs_consecuente = self.consecuente.simplificar2recompensa()
            if drs_antecedente.conds and drs_consecuente.conds:
                drs += dexpr(f'([][({str(drs_antecedente)} -> {str(drs_consecuente)})])')
            elif drs_antecedente.conds and not drs_consecuente.conds:
                drs += dexpr(f'([][({str(drs_antecedente)})])')
            elif not drs_antecedente.conds and drs_consecuente.conds:
                drs += dexpr(f'([][({str(drs_consecuente)})])')
        if self.antecedente is not None and self.consecuente is None:
            drs_antecedente = str(self.antecedente.simplificar2recompensa())
            drs += dexpr(f'([][({drs_antecedente})])')
        if self.antecedente is None and self.consecuente is not None:
            drs_consecuente = str(self.consecuente.simplificar2recompensa())
            drs += dexpr(f'([][({drs_consecuente})])')
        return drs.simplify()
    
    @staticmethod
    def get_fol(drs) -> str:
        """
        Convierte una expresión DRS a FOL (First-Order Logic).
        Si la expresion DRS no tiene condiciones, pero cuenta con elementos en 
        la lista de condiciones, se retorna la representación FOL de la primera
        condición. Si la lista de condiciones esta vacía, se retorna una cadena
        vacía.

        Args:
            drs (drt.DrtExpression): Expresión DRS a convertir.
        Returns:
            str: Expresión en lógica de primer orden (FOL).
        Raises:
            Exception: Si la conversión no es posible por una razon diferente a la falta de condiciones.   
        """
        try:
            return drs.fol()
        except Exception as e:
            if str(e) == "Cannot convert DRS with no conditions to FOL.":
                if drs.conds:
                    return Nodo.get_fol(drs.conds[0])
                else:
                    return ""
            else:
                raise e
        

class Estado2D:
    """
    Representa el estado del parser en un momento dado.

    Atributos:
        indice (int): Índice actual del estado.
        frase (str): Frase original a procesar.
        lista_palabras (list[str]): Lista de palabras tokenizadas de la frase.
        nodo (Nodo): Nodo asociado al estado para la representación lógica.
        lemas (list[str]): Lista de lemas extraídos de la frase.
    """
    def __init__(self, frases:List[str]) -> None:
        """
        Inicializa un estado con una frase dada y procesa sus tokens y lemas.
        Args:
            frase (str): Frase a procesar.
        """
        self.indice = 0
        assert(len(frases) == 2)
        self.frases = frases
        self.sep_index = None
        self.lista_palabras = self.obtener_tokens() 
        assert(self.sep_index is not None) # <= Asegurar que el separador de frases fue encontrado
        self.lemas = self.obtener_lemas()
        self.nodo1 = Nodo()
        self.nodo2 = Nodo()
        
    def __str__(self) -> str:
        """
        Retorna la representación en cadena del estado.
        Returns:
            str: Representación del estado con el índice, lista de palabras y la representación DRS simplificada.
        """
        msg = f'''
Índice: {self.indice}
Lista palabras: {self.lista_palabras}
DRS1:\n{self.nodo1.drs.pretty_format()}
DRS2:\n{self.nodo2.drs.pretty_format()}
'''
        return msg
    
    def obtener_tokens(self) -> list[stanza.models.common.doc.Token]:
        """
        Tokeniza la frase y retorna una lista de tokens.
        Returns:
            list[stanza.models.common.doc.Token]: Lista de tokens de la frase.
        """
        doc = list()
        frase = '. '.join(self.frases) + '.'
        list_len_tokens = list()
        for sentence in nlp(frase).sentences:
            tokens = sentence.tokens
            doc += tokens
            list_len_tokens.append(len(tokens))
        self.sep_index = list_len_tokens[0]
        return [x.to_dict()[0]['text'] for x in doc]      

    def obtener_lemas(self) -> list[stanza.models.common.doc.Token]:
        """
        Extrae los lemas de cada palabra en la frase.
        Returns:
            list[str]: Lista de lemas correspondientes a las palabras en la frase.
        """
        doc = list()
        frase = '. '.join(self.frases) + '.'
        for sentence in nlp(frase).sentences:
            doc += sentence.tokens
        
        lemas = list()
        for token in doc:
            # Manejar los casos si multiples lemmas estan presentes 
            lemmas = [entry.get('lemma') if entry.get('lemma') is not None else '<pad>' for entry in token.to_dict()]
            lemas += lemmas
        # Reemplazar punto (".") por "<eos>"
        lemas = ['<eos>' if x == '.' else x for x in lemas]
        return lemas

    def obtener_cadena(self) -> str:
        """
        Retorna la frase reconstruida a partir de los tokens.
        Returns:
            str: Representación de la frase.
        """
        oracion = ' '.join(self.lista_palabras)
        drs = self.nodo.simplificar()
        #cadena = f'\n{self.indice}\n{oracion}\n{drs}'
        cadena = oracion
        return cadena

    def get_nodo_indice(self) -> int:
        """
        Retorna el índice del nodo actual.
        Returns:
            int: Índice del nodo actual.
        """
        if self.indice < self.sep_index:
            return 0
        return 1

    def get_nodo(self) -> nltk.DrtExpression:
        nodo_indice = self.get_nodo_indice()
        if nodo_indice == 0:
            return self.nodo1
        else:
            return self.nodo2


class Embeddings(Box):
    """
    Representa el espacio de embeddings.

    Hereda de la clase `Box` de Gym, la cual define un espacio de observación con valores continuos.
    En este caso, los embeddings son representados en un espacio de dimensión (1, EMB_DIM).

    Atributos:
        low (float): Límite inferior del espacio (establecido en -∞).
        high (float): Límite superior del espacio (establecido en ∞).
        shape (tuple): Dimensión del espacio de embeddings (1, EMB_DIM).
        dtype (np.dtype): Tipo de dato de los embeddings (np.float32).
    """
    def __init__(self) -> None:
        """
        Inicializa un espacio de embeddings continuo sin restricciones de valores.

        Los valores de los embeddings pueden tomar cualquier valor real dentro de la representación
        de punto flotante.

        Inherits:
            Box: Clase de Gym que representa un espacio de valores continuos.
        """
        low = -np.inf
        high = np.inf
        shape = (1, EMB_DIM)
        dtype = np.float32
        super().__init__(low, high, shape, dtype, seed=None)


