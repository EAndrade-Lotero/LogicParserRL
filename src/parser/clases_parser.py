import re
import nltk
import numpy as np
from pandas import DataFrame
import nltk.sem.drt as drt

from pyprover import *
from pathlib import Path
from copy import deepcopy
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from nltk.sem.logic import LogicParser, Expression
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance

from src.parser.parser_auxiliary_classes import Estado, Nodo
from src.parser.interpreters import parser_interpreter

EMB_DIM = 1536

# Parse into DRS
dexpr = drt.DrtExpression.fromstring
# Parse into FOL
lp = LogicParser()


class Parser(Env):

    '''Clase del parser que se encarga de realizar las acciones sobre el DRS'''

    def __init__(
                self, 
                frase1: str, 
                frase2: str, 
                relacion: int,
                frase1_fol: str,
                frase2_fol: str, 
                max_turns: Optional[int]=100
            ) -> None:
        self.nombre_acciones = [
            "mover_derecha",
            "mover_izquierda",
            "incluir_predicado",
            "incluir_predicado_sin_ref",
            "enmascarar",
            "crear_drs_antecedente",
            "crear_drs_consecuente",
            "crear_drs_negacion",
            "subir_nivel"
        ]
        self.frase1 = frase1
        self.frase2 = frase2
        self.relacion = relacion
        self.frase1_fol = frase1_fol
        self.frase2_fol = frase2_fol
        self.estado = Estado([frase1, frase2])
        self.raiz = self.estado.nodo
        self.lista_palabras = deepcopy(self.estado.lista_palabras)
        self.debug = False
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(EMB_DIM,), dtype=np.float32
        )
        self.action_space = Discrete(len(self.nombre_acciones))
        self.turn = 0
        self.max_turns = max_turns
        
    def accion_to_index(self, accion:str) -> int:
        try:
            index = self.nombre_acciones.index(accion)
            return index
        except:
            raise Exception(f'Error: acción {accion} desconocida. Debe ser una de\n{self.nombre_acciones}')
    
    def reset(self) -> Estado:
        self.turn = 0
        self.estado = Estado(self.frase)
        self.raiz = self.estado.nodo
        observation = parser_interpreter({"Estado": self.estado, "Raiz": self.raiz})
        info = {}
        return observation, info 
    
    def render(self) -> None:
        print(self.raiz.simplificar().pretty_format())
        print(self.estado)

    def mover_derecha(self) -> None:
        if self.estado.indice < len(self.estado.lista_palabras) - 1:
            self.estado.indice +=  1
    
    def mover_izquierda(self) -> None:
        if self.estado.indice > 0:
            self.estado.indice -= 1
        
    def incluir_predicado(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != '[MASK]':
            self.estado.lista_palabras[self.estado.indice] = '[MASK]'  
            new_expresion = dexpr(f'([x], [{self.estado.lemas[self.estado.indice].upper()}(x)])')
            drs = self.estado.get_nodo()
            drs += new_expresion
            drs = drs.simplify()
    
    def incluir_predicado_sin_ref(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != '[MASK]':
            self.estado.lista_palabras[self.estado.indice] = '[MASK]'
            new_expresion = dexpr(f'([], [{self.estado.lemas[self.estado.indice].upper()}(x)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
        
    def enmascarar(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != '[MASK]':  
            self.estado.lista_palabras[self.estado.indice] = '[MASK]'    
       
    def crear_drs_antecedente(self) -> None:
        if self.estado.nodo.madre is None:
            nodo_antecedente = Nodo().crear_antecedente(self.raiz)
        else:
            nodo_antecedente = Nodo().crear_antecedente(self.estado.nodo)
        self.estado.nodo = nodo_antecedente
            
    def crear_drs_consecuente(self) -> None:
        if self.estado.nodo.madre is None:
            nodo_consecuente = Nodo().crear_consecuente(self.raiz)
        else:
            nodo_consecuente = Nodo().crear_consecuente(self.estado.nodo)
        self.estado.nodo = nodo_consecuente

    def crear_drs_negacion(self) -> None:
        if self.estado.nodo.madre is None:
            nodo_neg = Nodo().crear_negacion(self.raiz)
        else:
            nodo_neg = Nodo().crear_negacion(self.estado.nodo)
        self.estado.nodo = nodo_neg

    def subir_nivel(self) -> None:
        if self.estado.nodo.madre is not None:
            madre = self.estado.nodo.madre
            self.estado.nodo = madre
            self.estado.nodo.drs = self.estado.nodo.simplificar()
            self.estado.nodo.negacion = None
            if self.estado.nodo.antecedente is not None and self.estado.nodo.consecuente is not None:
                self.estado.nodo.antecedente = None
                self.estado.nodo.consecuente = None
             
    def obtener_recompensa(self, dict_referencias: dict[str, object]) -> int:
        if dict_referencias['done']:
            recompensa = self.obtener_recompensa_parsing_finalizado()
        else:
            recompensa = self.obtener_recompensa_parsing_no_finalizado(dict_referencias)
        return recompensa

    def obtener_done(self) -> bool:
        done = np.all([palabra=='[MASK]' for palabra in self.estado.lista_palabras])
        return bool(done)
        
    def step(self, accion:int) -> tuple[Estado, int, bool]:
        self.turn +=1
        dict_referencias = {'indice': self.estado.indice,
                            'palabra_accionada': self.estado.lista_palabras[self.estado.indice],
                            'len_lista_palabras': len(self.estado.lista_palabras),
                            'nombre_accion': self.nombre_acciones[accion],
                            'accion': accion}
        metodo = self.nombre_acciones[accion]
        metodo = f'self.{metodo}()'
        try:
            exec(metodo)
            dict_referencias['done'] = self.obtener_done()
            recompensa = self.obtener_recompensa(dict_referencias)
        except Exception as e:
            if self.debug: print(f"Accion erronea: {dict_referencias}")
            if self.debug: print(f"Error: {e}")
            dict_referencias['done'] = False
            recompensa = -1
        observation = parser_interpreter({"Estado": self.estado, "Raiz": self.raiz})
        truncated = self.turn > self.max_turns
        return observation, recompensa, dict_referencias['done'], truncated, dict_referencias
    
    def obtener_recompensa_parsing_no_finalizado(self, dict_referencias: dict[str, object]) -> int:
        # Penalizar acciones incorrectas
        if dict_referencias['accion'] == 'mover_izquierda' and dict_referencias['indice'] == 0:
            recompensa = -10
        elif dict_referencias['accion'] == 'mover_derecha' and dict_referencias['indice'] == dict_referencias['len_lista_palabras']-1:
            recompensa = -10
        elif dict_referencias['palabra_accionada'] == '[MASK]' and (dict_referencias['accion'] not in ['mover_derecha', 'mover_izquierda']):
            recompensa = -10
        elif dict_referencias['accion'] == 'subir_nivel' and self.estado.nodo.madre is None:
            recompensa = -10
        # Costo por acción
        else:
            recompensa = -1
        return recompensa

    def obtener_recompensa_parsing_finalizado(self) -> int:
        # Obtener la fórmula de la oración a partir del DRS
        premisas_fol = self.raiz.simplificar().fol()
        transformer = FormulaTransformer([premisas_fol, self.respuesta], debug=self.debug)
        premisas, conclusion = transformer.uniform()
        if self.debug:
            print('Premisas en NLTK:\n', premisas)
            print('Conclusion en NLTK:\n', conclusion)
        premisas = Parse2pyprover.parse(premisas)
        conclusion = Parse2pyprover.parse(conclusion)
        if self.debug:
            print('Premisas en Pyprover:\n', premisas)
            print('Conclusion en Pyprover:\n', conclusion)
        if self.debug:
            print(f'Evaluando si las premisas \n\t{premisas}\nimplican la conclusión\n\t{conclusion}')
        resultado = proves([premisas], conclusion)
        if self.debug:
            print(f'El resultado es {resultado}')
        if resultado:
            recompensa = 100
        else:
            recompensa = -100
        return recompensa
    



