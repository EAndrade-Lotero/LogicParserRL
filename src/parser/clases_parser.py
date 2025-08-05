import re
import nltk
import time
import stanza
import numpy as np
import nltk.sem.drt as drt

from pathlib import Path
from copy import deepcopy
from gymnasium import Env
from pandas import DataFrame
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from typing import List, Dict, Tuple, Optional
from nltk.sem.logic import LogicParser, Expression

from interpreters import parser_interpreter
from parser.parser_utils import FormulaTransformer
from parser.parser_auxiliary_classes import Nodo, Estado, Embeddings

from utils.variables import MASK, EMB_DIM

#Prover
# from nltk.inference import Prover9


# NLP pipeline for tokenization
nlp = stanza.Pipeline(
    lang='en', 
    processors='tokenize,pos,lemma', 
    verbose=False
)
# Parse into DRS
dexpr = drt.DrtExpression.fromstring
# Parse into FOL
lp = LogicParser()

# Paths
gram_folder = Path('../gramaticas/')

COUNT_ID = 2000


class Parser(Env):

    def __init__(
                self, 
                frase: str, 
                pregunta: str, 
                respuesta: str, 
                tipo_pregunta: str,
                max_turns: Optional[int]=20
            ) -> None:
        self.nombre_acciones = [
            "mover_derecha",
            "enmascarar",
            "incluir_sustantivo",
            "incluir_sustantivo_sin_ref",
            "crear_drs_antecedente",
            "crear_drs_consecuente",
            # "subir_nivel", # <= OJO, puede ser necesario
            # "mover_izquierda",
            # "incluir_sustantivo_plural",
            # "incluir_constante",
            # "convertir_plural",
            # "incluir_sujeto",
            # "incluir_verbo",
            # "incluir_objeto_dir",
            # "incluir_igualdad",
            # "incluir_relacion",
            # "incluir_relacion_negado",
            # "crear_drs_antecedente_focal",
            # "crear_drs_negacion",
        ]
        self.pregunta = pregunta
        self.respuesta = respuesta
        self.frase = frase
        self.tipo_pregunta = tipo_pregunta
        self.estado = Estado(self.frase)
        self.raiz = self.estado.nodo
        self.lista_palabras = deepcopy(self.estado.lista_palabras)
        self.unicidades = list()
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(EMB_DIM,), dtype=np.float32
        )
        self.action_space = Discrete(len(self.nombre_acciones))
        self.max_turns = max_turns
        self.max_recursion = 2
        self.turn = 0
        self.debug = False
        self.recompensa_parsing_finalizado = 50
        self.recompensa_frase_incorrecta = -10
        self.recompensa_accion_incorrecta = -10
        
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
        drs = self.raiz.simplificar2recompensa()
        print(drs.pretty_format())
        print(self.estado)
        
    def mover_derecha(self) -> None:
        if self.estado.indice < len(self.estado.lista_palabras) - 1:
            self.estado.indice +=  1
    
    def mover_izquierda(self) -> None:
        if self.estado.indice > 0:
            self.estado.indice -= 1
        
    def incluir_sustantivo(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK  
            new_expresion = dexpr(f'([x], [{self.estado.lemas[self.estado.indice].upper()}(x)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
            # Sube un nivel automáticamente al moverse a la derecha OJO!!!!!!!!!
            self.subir_nivel()
            self.mover_derecha()
   
    def incluir_sustantivo_sin_ref(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK
            new_expresion = dexpr(f'([], [{self.estado.lemas[self.estado.indice].upper()}(x)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
            # Sube un nivel automáticamente al moverse a la derecha OJO!!!!!!!!!
            self.subir_nivel()
            self.mover_derecha()
    
    def incluir_sustantivo_plural(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK  
            new_expresion = dexpr(f'([x, y], [{self.estado.lemas[self.estado.indice].upper()}(x), {self.estado.lemas[self.estado.indice].upper()}(y), -(x=y)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
    
    def incluir_constante(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK
            # drs_antecedente = f"([y], [{self.estado.lemas[self.estado.indice].upper()}(y)])"
            # drs_consecuente = "([], [(x=y)])"
            # drs_unicidad = dexpr(f'([][({drs_antecedente} -> {drs_consecuente})])') 
            # new_expresion = dexpr(f'([x], [{self.estado.lemas[self.estado.indice].upper()}(x), {drs_unicidad}])')           
            # drs_unicidad = dexpr(f'([x][{self.estado.lemas[self.estado.indice].upper()}(x), ({drs_antecedente} -> {drs_consecuente})])') 
            # new_expresion = dexpr(f'([], [{drs_unicidad}])')
            # formula = lp.parse(f"exists x.({self.estado.lemas[self.estado.indice].upper()}(x) & all y.({self.estado.lemas[self.estado.indice].upper()}(y) -> (x = y)))")
            # new_expresion = dexpr(f'([], [{formula}])')
            
            formula = lp.parse(f"exists x.({self.estado.lemas[self.estado.indice].upper()}(x) & all y.({self.estado.lemas[self.estado.indice].upper()}(y) -> (x = y)))")
            self.unicidades.append(formula)
            new_expresion = dexpr(f'([y], [{self.estado.lemas[self.estado.indice].upper()}(y)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()

    def convertir_plural(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK
            condiciones = self.estado.nodo.drs.conds
            referentes = self.estado.nodo.drs.get_refs()
            for ref in referentes:
                formula = []
                for cond in condiciones:
                    vars = [str(v) for v in cond.free()]
                    if str(ref) in vars:
                        condition = str(cond).replace(str(ref), f'{ref}1000')
                        formula.append(condition)
                new_expresion = f"([{ref}1000][{', '.join(formula)}, -({ref}={ref}1000)])"
                self.estado.nodo.drs += dexpr(new_expresion)
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()    

    def incluir_sujeto(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK      
            new_expresion = dexpr(f'([x],[{self.estado.lemas[self.estado.indice].upper()}(x),SUJETO(e,x)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
        
    def incluir_verbo(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:
            self.estado.lista_palabras[self.estado.indice] = MASK    
            new_expresion = dexpr(f'([e], [{self.estado.lemas[self.estado.indice].upper()}(e)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()

    def incluir_objeto_dir(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:  
            self.estado.lista_palabras[self.estado.indice] = MASK    
            new_expresion = dexpr(f'([y], [{self.estado.lemas[self.estado.indice].upper()}(y),OBJ_DIR(e,y)])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
    
    def enmascarar(self) -> None:
        if self.estado.lista_palabras[self.estado.indice] != MASK:  
            self.estado.lista_palabras[self.estado.indice] = MASK    
            self.mover_derecha()
    
    def incluir_igualdad(self) -> None:
        variables = self.estado.nodo.drs.get_refs()
        if len(variables) >= 2:
            variables = variables[-2:]
            new_expresion = dexpr(f'([], [{variables[0]}={variables[1]}])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
        else:
            raise Exception('No hay suficientes variables para incluir la igualdad.')

    def incluir_relacion(self) -> None:
        variables = self.estado.nodo.drs.get_refs().copy()
        
        if self.estado.nodo.madre is not None and self.estado.nodo.madre.var_focal is not None:
            variables.insert(0, self.estado.nodo.madre.var_focal)

        if self.estado.lista_palabras[self.estado.indice] != MASK and len(variables) >= 2:
            self.estado.lista_palabras[self.estado.indice] = MASK  
            predicado = self.estado.lemas[self.estado.indice].upper()
            variables = variables[-2:]
            new_expresion = dexpr(f'([], [{predicado}({variables[0]},{variables[1]})])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
        else:
            raise Exception('No hay suficientes variables para incluir la igualdad.')

    def incluir_relacion_negado(self) -> None:
        variables = self.estado.nodo.drs.get_refs()
        
        if self.estado.nodo.madre is not None and self.estado.nodo.madre.var_focal is not None:
            variables.insert(0, self.estado.nodo.madre.var_focal)
        
        if self.estado.lista_palabras[self.estado.indice] != MASK and len(variables) >= 2:  
            self.estado.lista_palabras[self.estado.indice] = MASK  
            predicado = self.estado.lemas[self.estado.indice].upper()
            variables = variables[-2:]
            new_expresion = dexpr(f'([], [-{predicado}({variables[0]},{variables[1]})])')
            self.estado.nodo.drs += new_expresion
            self.estado.nodo.drs = self.estado.nodo.drs.simplify()
    
    def crear_drs_antecedente_focal(self) -> None:
        if self.estado.nodo.nivel() < self.max_recursion:
            if self.estado.nodo.madre is None:
                nodo_antecedente = Nodo().crear_antecedente(self.raiz)
            else:
                nodo_antecedente = Nodo().crear_antecedente(self.estado.nodo)
            nodo_antecedente.madre.var_focal = f'x{np.random.randint(3000, 4000)}'       
            self.estado.nodo = nodo_antecedente
            new_expresion = dexpr(f'([{self.estado.nodo.madre.var_focal}], [])')
            self.estado.nodo.drs += new_expresion
        elif self.estado.nodo.nivel() == self.max_recursion:
            raise Exception(f'Intento de transgresión del nivel máximo de recursión ({self.estado.nodo.nivel()})')

    def crear_drs_antecedente(self) -> None:
        if self.estado.nodo.nivel() < self.max_recursion:
            if self.estado.nodo.antecedente is not None:
                raise Exception('Ya existe un antecedente en el nodo actual.')
            elif self.estado.nodo.madre is None:
                nodo_antecedente = Nodo().crear_antecedente(self.raiz)
            else:
                nodo_antecedente = Nodo().crear_antecedente(self.estado.nodo)
            self.estado.nodo = nodo_antecedente
        elif self.estado.nodo.nivel() == self.max_recursion:
            raise Exception(f'Intento de transgresión del nivel máximo de recursión ({self.estado.nodo.nivel()})')
            
    def crear_drs_consecuente(self) -> None:
        if self.estado.nodo.nivel() < self.max_recursion:
            if self.estado.nodo.consecuente is None:
                if self.estado.nodo.madre is None:
                    nodo_consecuente = Nodo().crear_consecuente(self.raiz)
                else:
                    nodo_consecuente = Nodo().crear_consecuente(self.estado.nodo)
                self.estado.nodo = nodo_consecuente
            else:
                self.estado.nodo = self.estado.nodo.consecuente
        elif self.estado.nodo.nivel() == self.max_recursion:
            raise Exception(f'Intento de transgresión del nivel máximo de recursión ({self.estado.nodo.nivel()})')

    def crear_drs_negacion(self) -> None:
        if self.estado.nodo.nivel() < self.max_recursion:
            if self.estado.nodo.madre is None:
                nodo_neg = Nodo().crear_negacion(self.raiz)
            else:
                nodo_neg = Nodo().crear_negacion(self.estado.nodo)
            self.estado.nodo = nodo_neg
        elif self.estado.nodo.nivel() == self.max_recursion:
            raise Exception(f'Intento de transgresión del nivel máximo de recursión ({self.estado.nodo.nivel()})')

    def subir_nivel(self) -> None:
        if self.estado.nodo.madre is not None:
            madre = self.estado.nodo.madre
            self.estado.nodo = madre
            # self.estado.nodo.drs = self.estado.nodo.simplificar()
            # self.estado.nodo.negacion = None
            # if self.estado.nodo.antecedente is not None and self.estado.nodo.consecuente is not None:
            #     self.estado.nodo.antecedente = None
            #     self.estado.nodo.consecuente = None
    
    def obtener_recompensa(self, dict_referencias: dict[str, object]) -> int:
        if dict_referencias['done']:
            recompensa = self.obtener_recompensa_parsing_finalizado()
        else:
            recompensa = self.obtener_recompensa_parsing_no_finalizado(dict_referencias)
        return recompensa
    
    def obtener_recompensa_parsing_no_finalizado(self, dict_referencias: dict[str, object]) -> int:
        if dict_referencias['accion'] == 'mover_izquierda' and dict_referencias['indice'] == 0:
            recompensa = -10
        elif dict_referencias['accion'] == 'mover_derecha' and dict_referencias['indice'] == dict_referencias['len_lista_palabras']-1:
            recompensa = -10
        elif dict_referencias['palabra_accionada'] == MASK and (dict_referencias['accion'] not in ['mover_derecha', 'mover_izquierda']):
            recompensa = -10
        elif dict_referencias['accion'] == 'subir_nivel' and self.estado.nodo.madre is None:
            recompensa = -10
        # elif self.turn > self.max_turns:
        #     recompensa = -100
        else:
            recompensa = -1
        return recompensa

    def obtener_recompensa_parsing_finalizado(self) -> int:
        # Usamos la DRS y la pregunta para generar candidatos
        candidatos, dict_fol = self.encontrar_candidatos()
        # print(f'Los candidatos a probar son:\n\t{[str(x) for x in candidatos]}')
        respuestas_obtenidas = []
        for candidato in candidatos:
            res = self.evaluar_candidato(candidato, dict_fol)
            if self.debug:
                print(f'Evaluando al candidato {candidato}')
                print(f'El resultado es {res}')
            if res == 'UNSAT':
                respuestas_obtenidas.append(candidato)
        respuestas_obtenidas = [str(respuesta) for respuesta in respuestas_obtenidas]
        if self.debug:
            print(respuestas_obtenidas)
        if self.respuesta in respuestas_obtenidas:
            recompensa = 100
        else:
            recompensa = -100
        return recompensa

    def obtener_recompensa_parsing_finalizado(self) -> int:
        # Fórmula creada por el agente
        drs_raiz = self.raiz.simplificar2recompensa()
        premisa1 = str(Nodo.get_fol(drs_raiz))
        # Fórmula correcta
        fol_correcta = str(FormulaTransformer(self.frase_recompensa).formulas[0])
        # Verificar si fórmula creada es igual a la correcta
        if premisa1 == fol_correcta:
            recompensa = self.recompensa_parsing_finalizado
        else:
            recompensa = self.recompensa_frase_incorrecta
        return recompensa

    def obtener_recompensa_parsing_no_finalizado(self, dict_referencias: dict[str, object]) -> int:
        acciones_sobre_mask = [
            "mover_derecha",
            "mover_izquierda",
            "incluir_igualdad",
            "subir_nivel"
        ]
        acciones_crear_drs = [
            "crear_drs_antecedente",
            "crear_drs_consecuente",
            "crear_drs_negacion"
        ]
        acciones_en_palabra = [
            "enmascarar",
            "incluir_sustantivo",
            "incluir_sustantivo_sin_ref",
        ]
        if self.debug: print("\nEvaluacion de recomepensas")
        # if self.turn > self.max_turns:
        #     recompensa = -self.recompensa_parsing_finalizado
        if dict_referencias['palabra_accionada'] == MASK and dict_referencias['nombre_accion'] not in acciones_sobre_mask:
            recompensa = self.recompensa_accion_incorrecta
        elif dict_referencias['nombre_accion'] in acciones_sobre_mask:
            if dict_referencias['nombre_accion'] == "mover_izquierda":
                if dict_referencias['indice'] == 0:
                    recompensa = self.recompensa_accion_incorrecta
                else:
                    recompensa = self.recompensa_accion_correcta
            elif dict_referencias['nombre_accion'] == "mover_derecha":
                if dict_referencias['indice'] == dict_referencias['len_lista_palabras']-1:
                    recompensa = self.recompensa_accion_incorrecta
                else:
                    recompensa = self.recompensa_accion_correcta
            elif dict_referencias['nombre_accion'] == 'subir_nivel':
                if self.estado.nodo.madre is None:
                    recompensa = self.recompensa_accion_incorrecta
                else:
                    recompensa = self.recompensa_accion_correcta
            else:
                recompensa = self.get_recompensa_guided_reward(dict_referencias)
        elif dict_referencias['nombre_accion'] in acciones_crear_drs:
            nodo_level = self.estado.nodo.nivel()
            if dict_referencias['nombre_accion'] == 'crear_drs_negacion' and nodo_level >= self.max_recursion:
                recompensa = self.recompensa_accion_incorrecta
            elif nodo_level > 1:
                recompensa = self.recompensa_accion_incorrecta
            else:
                recompensa = self.get_recompensa_guided_reward(dict_referencias)
        elif dict_referencias['nombre_accion'] in acciones_en_palabra:
            recompensa = self.get_recompensa_guided_reward(dict_referencias)
        else:
            raise Exception(f'Error: Acción {dict_referencias["nombre_accion"]} no clasificada.') 
        if self.debug: 
            print(f"La recompensa a retornar con la accion {dict_referencias['nombre_accion']} es {recompensa}")
        return recompensa

    def get_recompensa_guided_reward(self, dict_referencias:Dict[str, any]) -> float:
        return 0

    def obtener_done(self) -> bool:
        done = np.all([palabra == MASK for palabra in self.estado.lista_palabras])
        return bool(done)
        
    def step(self, accion:int) -> tuple[Estado, int, bool]:
        self.turn +=1
        dict_referencias = {'indice': self.estado.indice,
                            'palabra_accionada': self.estado.lista_palabras[self.estado.indice],
                            'len_lista_palabras': len(self.estado.lista_palabras),
                            'nombre_accion': self.nombre_acciones[accion],
                            'accion': accion}
        try:
            nombre_metodo = self.nombre_acciones[accion]
            metodo = getattr(self, nombre_metodo)
            metodo()
            dict_referencias['done'] = self.obtener_done()
            recompensa = self.obtener_recompensa(dict_referencias)
        except Exception as e:
            if self.debug: print(f"Accion erronea: {dict_referencias}")
            if self.debug: print(f"Error: {e}")
            dict_referencias['done'] = False
            recompensa = self.recompensa_frase_incorrecta
            # raise Exception(f"Accion erronea: {dict_referencias}. Error: {e}")

        observation = parser_interpreter({"Estado": self.estado, "Raiz": self.raiz})
        truncated = self.turn >= self.max_turns
        return observation, recompensa, dict_referencias['done'], truncated, dict_referencias


class Parser2(Parser):
    def __init__(self, df):
        self.df = df
        frase, pregunta, respuesta, tipo_pregunta = self.random_init()
        super().__init__(frase, pregunta, respuesta, tipo_pregunta)

    def random_init(self):
        idx = np.random.randint(0,self.df.shape[0])
        frase, pregunta, respuesta, tipo_pregunta = self.df.loc[idx]
        return frase, pregunta, respuesta, tipo_pregunta
    
    def reset(self):
        self.frase, self.pregunta, self.respuesta, self.tipo_pregunta = self.random_init()
        return super().reset()