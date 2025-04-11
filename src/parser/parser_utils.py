import re
import nltk

from pyprover import *
from typing import List, Dict
from nltk.sem.logic import LogicParser, Expression

class Parse2pyprover:
    """
    Clase para convertir expresiones lógicas de NLTK en una representacion compatible con Pyprover.
    """
    verbose = False
    
    @staticmethod
    def set_verbosemode(value:bool) -> None:
        """
        Determina el modo verbose de la clase.
        Args:
            value (bool): Valor para el modo verbose.
        """
        Parse2pyprover.verbose = value
    
    @staticmethod
    def parse(formula:nltk.sem.Expression) -> nltk.sem.Expression:
        """
        Convierte una expresión lógica de NLTK en una expresión lógica compatible con Pyprover.
        Args:
            formula (nltk.sem.Expression): Expresión lógica a convertir en formato NLTK.
        Returns:
            nltk.sem.Expression: Expresión lógica convertida en formato Pyprover.
        """
        Parse2pyprover.variables_a_terms(formula)
        cadena = str(formula)
        cadena = re.sub(r'all ', r'∀', cadena)
        cadena = Parse2pyprover.reemplazar_existencial(cadena)
        if Parse2pyprover.verbose: print(f"Cadena con existenciales reemplazados:\n{cadena}")
        pyprover_exp = expr(cadena)
        return pyprover_exp

    @staticmethod
    def variables_a_terms(formula:nltk.sem.Expression) -> None:
        """
        Reemplaza las variables en una expresión lógica por términos en PyProver.
        Args:
            formula (nltk.sem.Expression): Expresión lógica.
        """
        variables = Parse2pyprover.encontrar_variables(formula)
        terms(' '.join(variables))

    @staticmethod
    def reemplazar_existencial(formula:str) -> str:
        """
        Reemplaza los cuantificadores existenciales en una expresión lógica por la notación de PyProver.
        Args:
            formula (str): Expresión lógica en formato de cadena.
        Returns:
            str: Expresión lógica con los cuantificadores existenciales reemplazados.
        """
        pattern_exists = re.compile('exists')
        if not pattern_exists.search(formula):
            return formula
        for m in pattern_exists.finditer(formula):
            span_init = m.span()[0]
            span_start = m.span()[1] + 1
            span_end = re.search(r'\.', formula).span()[0]
            span_nuevo = re.search(r'\.', formula).span()[1]
            lista_variables = formula[span_start:span_end].split(' ')
            lista_variables = [f'∃{v}.' for v in lista_variables]
            existenciales = ' '.join(lista_variables)
            nueva_formula = formula[:span_init] + existenciales + formula[span_nuevo:]
            if Parse2pyprover.verbose:
                print(nueva_formula)
            break
        return nueva_formula

    @staticmethod
    def encontrar_variables(expresion:nltk.sem.Expression) -> List[str]:
        """
        Encuentra las variables presentes en una expresión lógica.
        Args:
            expresion (nltk.sem.Expression): Expresión lógica.
        Returns:
            List[str]: Lista de variables presentes en la expresión lógica.
        """
        tipo = Parse2pyprover.obtener_type(expresion)
        if Parse2pyprover.verbose: print(f'Estoy chequeando {expresion} de tipo {tipo}')
        if tipo in ['ExistsExpression', 'AllExpression']:
            var = expresion.variable.name
            variables = [var] + Parse2pyprover.encontrar_variables(expresion.term)
            return list(set(variables))
        elif tipo in ['AndExpression', 'OrExpression', 'ImpExpression']:
            variables1 = Parse2pyprover.encontrar_variables(expresion.first)
            variables2 = Parse2pyprover.encontrar_variables(expresion.second)
            variables = variables1 + variables2
            return list(set(variables))
        elif tipo in ['NegatedExpression']:
            variables = Parse2pyprover.encontrar_variables(expresion.term)
            return list(set(variables))
        elif tipo in ['ApplicationExpression', 'EqualityExpression']:
            variables = [v.name for v in expresion.variables()]
            return list(set(variables))
        elif tipo in ['ConstantExpression']:
            return list()
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {type(expresion)}')

    @staticmethod            
    def obtener_type(objeto):
        """
        Obtiene el tipo de una expresión lógica de manera legible.
        Args:
            objeto: Expresión lógica.
        Returns:
            str: Tipo de la expresión lógica.
        """
        c = str(type(objeto))
        return c.split('.')[-1][:-2]


class FormulaTransformer:
    """
    Clase para transformar fórmulas lógicas de NLTK a un formato compatible 
    con la estructura de FOLIO.
    """

    def __init__(self, formulas, debug=False):
        """
        Inicializa la clase FormulaTransformer con una fórmula o una lista de fórmulas.
        Args:
            formulas (str or list): Fórmula o lista de fórmulas a transformar.
            debug (bool): Modo verbose.
        """
        # Ensure the formulas are stored as a list of parsed formulas
        if isinstance(formulas, str):  # Single formula as a string
            self.formulas = [FormulaTransformer.translate(formulas)]
        elif isinstance(formulas, list):  # List of formulas (strings or parsed)
            self.formulas = [FormulaTransformer.translate(f) if isinstance(f, str) else f for f in formulas]
        else:  # Single formula, already parsed
            self.formulas = [formulas]
        self.debug = debug
    
    @staticmethod
    def translate(fol_expression):
        """
        Traduce una cadena de expresión FOL dada a la sintaxis de NLTK.
        Args:
            fol_expression (str): Cadena de expresión FOL.
        Returns:
            Expression: Expresión lógica en formato NLTK
        """
        # Replace specific operators with NLTK-compatible symbols
        translation = (
            fol_expression
            .replace('∃', 'exists ')
            .replace('∀', 'all ')
            .replace('∧', '&')
            .replace('∨', '|')
            .replace('¬', '-')
            .replace('→', '->')
            .replace('⊕', 'or') # Not define in NLTK
            .replace('=', '==')
        )
        return FormulaTransformer.upper_cons(lp.parse(translation))
    
    @staticmethod
    def upper_cons(formula):
        """
        Transforma una fórmula lógica dada convirtiendo los nombres de los 
        predicados a mayúsculas y aplicando la transformación de manera 
        recursiva a las subfórmulas basándose en su tipo.
        Args:
            formula: La fórmula lógica a transformar.
        Returns:
            Expression: La fórmula lógica transformada con los nombres de los predicados en mayúsculas.
        Raises:
            Exception: Si el tipo de la fórmula es desconocido.
        """
        tipo = obtener_type(formula)
        if tipo == 'ApplicationExpression':
            # En el caso de existir una relacion en la formula REL(x,y)
            if obtener_type(formula.function) == 'ApplicationExpression':
                predicado = str(formula.function.function.variable).upper()
                argumentos = ",".join([str(x) for x in formula.args])
            else:
                predicado = str(formula.function.variable).upper()
                argumentos = ','.join([str(x) for x in formula.args])
            formula = lp.parse(f"{predicado}({argumentos})")
            return formula
        elif tipo in ['ExistsExpression']:
            new_term = FormulaTransformer.upper_cons(formula.term)
            return lp.parse(f'exists {formula.variable}.({new_term})')
        elif tipo in ['AllExpression']:
            new_term = FormulaTransformer.upper_cons(formula.term)
            return lp.parse(f'all {formula.variable}.({new_term})')
        elif tipo in ['NegatedExpression']:
            new_term = FormulaTransformer.upper_cons(formula.term)
            return lp.parse(f'-{new_term}')
        elif tipo in ['AndExpression']:
            new_first = FormulaTransformer.upper_cons(formula.first)
            new_second = FormulaTransformer.upper_cons(formula.second)
            return lp.parse(f'({new_first} & {new_second})')
        elif tipo in ['OrExpression']:
            new_first = FormulaTransformer.upper_cons(formula.first)
            new_second = FormulaTransformer.upper_cons(formula.second)
            return lp.parse(f'({new_first} | {new_second})')
        elif tipo in ['ImpExpression']:
            new_first = FormulaTransformer.upper_cons(formula.first)
            new_second = FormulaTransformer.upper_cons(formula.second)
            return lp.parse(f'({new_first} -> {new_second})')
        else:
            raise Exception(f'Unknown type {tipo}')
    
    @staticmethod
    def const2exist(formula: Expression) -> Expression:
        """
        Transforma las constantes en una fórmula en cuantificadores existenciales.
        Args:
            formula (Expression): Fórmula lógica a transformar.
        Returns:
            Expression: Fórmula lógica con las constantes transformadas en cuantificadores existencia.
        """
        tipo = obtener_type(formula)
        if tipo == 'ApplicationExpression':
            for argument in formula.args:
                if obtener_type(argument) == 'ConstantExpression':
                    formula_string = str(formula)
                    new_string = formula_string.replace(str(argument), 'x1000')
                    new_string = f'exists x1000.({str(argument).upper()}(x1000) & {new_string})'
                    formula = lp.parse(new_string)
            return formula
        elif tipo in ['ExistsExpression']:
            new_term = FormulaTransformer.const2exist(formula.term)
            return lp.parse(f'exists {formula.variable}.({new_term})')
        elif tipo in ['AllExpression']:
            new_term = FormulaTransformer.const2exist(formula.term)
            return lp.parse(f'all {formula.variable}.({new_term})')
        elif tipo in ['AndExpression']:
            new_first = FormulaTransformer.const2exist(formula.first)
            new_second = FormulaTransformer.const2exist(formula.second)
            return lp.parse(f'({new_first} & {new_second})')
        elif tipo in ['EqualityExpression']:
            return lp.parse(f'({formula.first} = {formula.second})')
        elif tipo in ['OrExpression']:
            new_first = FormulaTransformer.const2exist(formula.first)
            new_second = FormulaTransformer.const2exist(formula.second)
            return lp.parse(f'({new_first} | {new_second})')
        elif tipo in ['ImpExpression']:
            new_first = FormulaTransformer.const2exist(formula.first)
            new_second = FormulaTransformer.const2exist(formula.second)
            return lp.parse(f'({new_first} -> {new_second})')
        else:
            raise Exception(f'Unknown type {tipo}')
        
    @staticmethod
    def complexity_score(word:str) -> int:
        """
        Calcular la complejidad de una palabra (proxy: longitud de la palabra).
        Args:
            word (str): Palabra a evaluar.
        Returns:
            int: Complejidad de la palabra.
        """
        return len(word)
    
    @staticmethod
    def similarity(a:str, b:str) -> float:
        """
        Calcula la similitud entre dos cadenas.
        Args:
            a (str): Primera cadena.
            b (str): Segunda cadena.
        Returns:
            float: Similitud entre las cadenas.
        """
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def most_complex_transform(self, predicados: List[str]) -> Dict[str, str]:
        """
        Genera un diccionario de transformación que mapea predicados a la versión más compleja.
        Args:
            predicados (List[str]): Lista de predicados.
        Returns:
            Dict[str, str]: Diccionario de transformación.
        """
        transformation = {}
        processed = set()
        if self.debug: print("Creacion del diccionario de transformacion")
        for i, word in enumerate(predicados):
            if self.debug: print(f"\nPalabra a procesar: {word}, Complejidad={self.complexity_score(word)}\nPalablras procesadas> {processed}")
            if word in processed:
                continue
            most_complex = word
            for j, other_word in enumerate(predicados):
                if i != j and self.similarity(word, other_word) > 0.5:  # Similarity threshold
                    if self.debug: print(f"Palabra con silimitud: {other_word}, similitud={self.similarity(word, other_word)}, Complejidad={self.complexity_score(other_word)}")
                    if self.complexity_score(other_word) > self.complexity_score(most_complex):
                        most_complex = other_word
                    else:
                        processed.add(word)
                        word = other_word
                    processed.add(other_word)
            transformation[word ] = most_complex
            processed.add(word)
        return transformation
    
    def reemplazar(self, lista_formulas: List[Expression], transformation_dict: Dict[str, str]) -> List[Expression]:
        """
        Remplaza los predicados en las fórmulas basándose en un diccionario de transformación.
        Args:
            lista_formulas (List[Expression]): Lista de fórmulas a transformar.
            transformation_dict (Dict[str, str]): Diccionario de transformación.
        Returns:
            List[Expression]: Lista de fórmulas transformadas.
        """
        new_list = []
        if self.debug: print("\n\nRemplazo de candidatos en las formulas")
        for formula in lista_formulas:
            formula_string = str(formula)
            if self.debug: print(f"Formula a estandarizar: {formula_string}")
            for x, y in transformation_dict.items():
                if self.debug: print(f"Items de trasnformacion: {x}>{y}")
                formula_string = formula_string.replace(x+"(", y+"(")
                if self.debug: print(f"Formla cambio x>y : {formula_string}")
            if self.debug: print(f"\nFormula final >> {formula_string}\n")
            formula = lp.parse(formula_string)
            new_list.append(formula)
        return new_list
    
    def uniform(self) -> List[Expression]:
        """
        Aplica la transformación uniforme a las fórmulas almacenadas.
        Returns:
            List[Expression]: Lista de fórmulas transformadas.
        """
        self.formulas = [self.const2exist(f) for f in self.formulas]
        predicados = []
        for formula in self.formulas:
            _, predicados_ = obtener_vocabulario(formula)
            predicados += [str(p) for p in predicados_]
            if self.debug: print(f"Formula >> {formula}")
        if self.debug: print(f"predicados >> {predicados}\n")
        transformation_dict = self.most_complex_transform(predicados)
        if self.debug: print(f"\nDiccionario de estandarizacion >> {transformation_dict}")
        self.formulas = self.reemplazar(self.formulas, transformation_dict)
        return self.formulas