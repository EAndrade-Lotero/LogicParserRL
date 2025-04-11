import re
import nltk
import stanza

from typing import List
from pathlib import Path
from nltk.tree import Tree
from nltk.grammar import FeatureGrammar
from nltk.parse import FeatureEarleyChartParser

from config.config import PATHS

nlp = stanza.Pipeline(
    lang='es', 
    processors='tokenize,pos,lemma', 
    verbose=False
)

class ParserUtils:

    @staticmethod
    def arbol_sin_caracteristicas(s:str) -> Tree:
        '''
        Toma un árbol de una cadena, de la cual elimina las características
        que se encuentran entre paréntesis cuadrados, y devuelve un Tree de nltk.
        Input:
            - s, una cadena con un árbol en representación plana
        Output:
            - un árbol de la clase Tree de la librería nltk
        '''
        s = s.replace('[', '{')
        s = s.replace(']', '}')
        s = re.sub('{.*?}', '', s)
        try:
            arbol = Tree.fromstring(s)
        except:
            s = re.sub(',.*?>}', '', s)  
            arbol = Tree.fromstring(s)
        return arbol

    @staticmethod
    def parsear(tokens:list, parser:FeatureEarleyChartParser, verbose=False) -> Tree:
        '''
        Toma una lista de tokens y devuelve el árbol de análisis
        usando el parser suministrado.
        Input:
            - toekns, una lista de cadenas con una oración
            - parser, un parser FeatureEarleyChartParser de nltk
            - verbose, booleano para imprimir información
        Output:
            - un árbol de la clase Tree de la librería nltk o None
        '''
        if verbose:
            print(f'Haciendo el parsing de la oración:\n\n\t{" ".join(tokens)}\n')
        trees = parser.parse(tokens)
        arboles = []
        for t in trees:
            if verbose:
                print(f'El árbol lineal obtenido es:\n\n\t{t}\n')
            return ParserUtils.arbol_sin_caracteristicas(str(t))
        if verbose:
            print('¡El parser no produjo ningún árbol!')
        return None

    @staticmethod
    def obtener_formula(tokens:list, parser:FeatureEarleyChartParser, clausura:bool=False) -> nltk.sem.logic:
        '''
        Toma una lista de tokens y devuelve su representación lógica.
        Input:
            - toekns, una lista de cadenas con una oración.
            - parser, un parser FeatureEarleyChartParser de nltk.
            - clausura, un booleano para devolver la fórmula clausurada o no.
        Output:
            - formula clausurada
        '''
        trees = parser.parse(tokens)
        for t in trees:
            formula = t.label().get('SEM')
            return formula
        return None
    

class CrearReglas:

    def __init__(
                self, 
                nouns:List[str], 
                adjs:List[str], 
                verbs:List[str], 
            ) -> None:
        self.nouns = nouns
        self.adjs = adjs
        self.verbs = verbs
        # Creamos el parser
        self.parser = self.crear_parser()

    def reglas_gramaticales(self) -> str:
        # Definimos la gramática
        grammar_folder = PATHS["grammar_folder"]
        grammar_file = Path(grammar_folder, "reglas_gramaticales.txt")
        with open(grammar_file, "r") as f:
            reglas_gramaticales = f.read()
        return reglas_gramaticales

    def reglas_nouns(self) -> str:
        # Definimos los sustantivos        
        reglas_nouns = list()
        for noun in self.nouns:
            l_noun = self.get_predicate(noun)
            regla_noun = fr"N[SEM=<\x.{l_noun}(x)>] -> '{noun}'"
            reglas_nouns.append(regla_noun)
        return reglas_nouns

    def reglas_verbs(self) -> str:
        # Definimos los verbos
        reglas_verbs = list()
        for verb in self.verbs:
            l_verb = self.get_predicate(verb)
            regla_verb = fr"VI[SEM=<\x.{l_verb}(x)>] -> '{verb}'"
            reglas_verbs.append(regla_verb)
        return reglas_verbs

    def reglas_adjs(self) -> str:
        # Definimos los adjetivos
        reglas_adjs = list()
        for adj in self.adjs:
            l_adj = self.get_predicate(adj)
            regla_adj = fr"ADJ[SEM=<\X.\x.(X(x) & {l_adj}(x))>] -> '{adj}'"
            reglas_adjs.append(regla_adj)
        return reglas_adjs

    def get_predicate(self, word: str) -> str:
        # Definimos los verbos
        doc = nlp(word)
        for sentence in doc.sentences:
            for word in sentence.words:
                l_word = word.lemma
        return l_word.upper()

    def crear_parser(self) -> FeatureEarleyChartParser:
        # Definimos las reglas de la gramática
        reglas_gramaticales = self.reglas_gramaticales()
        reglas_nouns = self.reglas_nouns()
        reglas_verbs = self.reglas_verbs()
        reglas_adjs = self.reglas_adjs()

        # Unimos todas las reglas
        reglas = "\n".join([reglas_gramaticales] + reglas_nouns + reglas_verbs + reglas_adjs)

        # Creamos la gramática
        grammar = FeatureGrammar.fromstring(reglas)

        # Creamos el parser
        parser = FeatureEarleyChartParser(grammar)
        
        return parser

    def to_fol(self, oracion: str) -> str:
        oracion_ = self.preprocesar(oracion)
        fol = ParserUtils.obtener_formula(oracion_, self.parser)
        if fol is not None:
            fol = fol.simplify()
            return fol
        else:
            return None
        
    def preprocesar(self, oracion: str) -> List[str]:
        oracion = oracion.strip()
        lista_palabras = oracion.split()
        if lista_palabras[0] == "no":
            lista_palabras = ['no_'] + lista_palabras[1:]
        return lista_palabras