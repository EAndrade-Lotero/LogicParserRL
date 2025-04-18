import nltk
import numpy as np

from nltk.sem.logic import LogicParser
from typing import (
	List, Optional, Tuple
)

from logic.logClases import *
from logic.logUtils import LogUtils

class ToPropositionalLogic:
     
    def __init__(self) -> None:
        self.parser = LogicParser()
        self.debug = False
        self.modelo_lp = Modelo()

    def parse(self, sentence:str) -> str:
        if isinstance(sentence, str):
            sentence_lp = self.parser.parse(sentence)
        elif not isinstance(sentence, Expression):
            raise Exception(f'Error: Expected {sentence} to be of type either string or nltk.sem.logic.Expression')
        else:
            sentence_lp = sentence
        assert(len(sentence_lp.free()) == 0), f'Fórmula con variables libres: {sentence_lp}\n\{sentence_lp.free()}'
        self.modelo_lp.poblar_con(sentence_lp)
        formula_fundamentada = self.modelo_lp.fundamentar(sentence_lp)
        formula_lp = self.modelo_lp.codificar_lp(formula_fundamentada)
        if self.debug:
            print(f'\n\nLa oración inicial es:\n{sentence}')
            print('El modelo queda:')
            print(self.modelo_lp)
            print(f'La fórmula fundamentada es:\n{formula_fundamentada}')
            print(f'La fórmula codificada es:\n{formula_lp}')
        return formula_lp

    def clases_no_vacias(self, sentence:str) -> str:
        sentence_lp = self.parser.parse(sentence)
        afirmacion_existencial = LogUtils.predicados_a_existenciales(sentence_lp)         
        afirmacion_existencial = LogUtils.existenciales_a_constantes(afirmacion_existencial)
        if afirmacion_existencial is None:
            formula_clases_no_vacias = sentence
        else:
            formula_clases_no_vacias = f'({sentence_lp} & {afirmacion_existencial})'
        return formula_clases_no_vacias


class ToNumeric:

    def __init__(self, clausal:List[List[str]]) -> None:
        self.clausal = clausal
        self.crear_vocab()

    def solo_atomo(self, literal:str) -> str:
        '''
        Convierte un literal a su forma atómica.
        Input:
            - literal, un string
        Output:
            - literal, un string
        '''
        if literal[0] == '-':
            return literal[1:]
        else:
            return literal

    def no_negaciones(self) -> List[List[str]]:
        '''
        Convierte una lista de listas de literales a una lista de listas de literales sin negaciones.
        Input:
            - clausal, una lista de listas de literales
        Output:
            - clausal_no_negaciones, una lista de listas de literales sin negaciones
        '''
        clausal_no_negaciones = list()
        for C in self.clausal:
            C_ = [self.solo_atomo(literal) for literal in C]
            clausal_no_negaciones.append(C_)
        return clausal_no_negaciones
    
    def crear_vocab(self) -> List[str]:
        '''
        Crea el vocabulario a partir de la lista de listas de literales.
        Input:
            - clausal, una lista de listas de literales
        Output:
            - vocab, una lista de letras proposicionales
        '''
        clausal_no_negaciones = self.no_negaciones()
        self.itos = ['<PAD>']
        self.stoi = {'<PAD>': 0}	
        for sentence in clausal_no_negaciones:
            for token in sentence:
                if token not in self.itos:
                    self.stoi[token] = len(self.itos)
                    self.itos.append(token)

    def como_literal(self, l:str) -> Tuple[str, str]:
        if '-' in l:
            return '-', l[1:]
        else:
            return '', l

    def mostrar_como_entero(self, literal):
        neg, atomo = self.como_literal(literal)
        if neg == '-':
            return -1 * self.stoi[atomo]
        else:
            return self.stoi[atomo]

    def to_numeric(self, clausal:List[List[str]]) -> List[List[int]]:
        '''
        Convierte una lista de listas de literales a una lista de listas de enteros.
        Input:
            - clausal, una lista de listas de literales
        Output:
            - clausal_num, una lista de listas de enteros
        '''
        clausal_num = list()
        for C in clausal:
            C_ = [self.mostrar_como_entero(literal) for literal in C]
            clausal_num.append(C_)
        return clausal_num        

    def from_numeric(self, clausal:List[List[int]]) -> List[List[str]]:
        '''
        Convierte una lista de listas de enteros a una lista de listas de literales.
        Input:
            - clausal, una lista de listas de enteros
        Output:
            - clausal_str, una lista de listas de literales
        '''
        clausal_str = list()
        for C in clausal:
            C_ = [self.itos[abs(literal)] if literal >= 0 else '-' + self.itos[abs(literal)] for literal in C]
            clausal_str.append(C_)
        return clausal_str
    
    def literal(self, number:int) -> str:
        '''
        Convierte un número a un literal.
        Input:
            - number, un número entero
        Output:
            - literal, un string
        '''
        if number < 0:
            return '-' + self.itos[abs(number)]
        else:
            return self.itos[number] 


class Modelo:
    '''
    Contendor del modelo de discurso.
    '''
    def __init__(self, formula=None):
        self.entidades = {}
        self.predicados = []
        self.vocabulario = []
        self.descriptor = None
        self.nltk_log_parser = nltk.sem.logic.LogicParser()
        if formula is not None:
            s = self.nltk_log_parser.parse(formula)
            self.poblar_con(s)

    def poblar_con(self, expresion:nltk.sem.logic):
        '''
        Toma una fórmula y extrae los individuos allí representados. 
        '''
        constantes, predicados = LogUtils.obtener_vocabulario(expresion)
        for p in predicados:
            if not p.en_conjunto(self.predicados):
                self.predicados.append(p)
       # Creamos las constantes
        for c in constantes:
            self.nueva_entidad(tipo=c.tipo, nombre=c.nombre)
        self.actualizar()

    def nueva_entidad(self, tipo:str, nombre:str):
        '''
        Crea una nueva entidad en la situación.
        '''
        try:
            n = len(self.entidades[tipo])
            nombres_previos = [c.nombre for c in self.entidades[tipo]]
            if nombre in nombres_previos:
                #print(f'¡Entidad ya existente! No se creó una nueva entidad. ({nombre})')
                pass
            else:
                self.entidades[tipo].append(Constante(tipo, nombre))
        except:
            n = 0
            self.entidades[tipo] = [Constante(tipo, nombre)]
        self.actualizar()

    def actualizar(self):
        '''
        Actualiza el vocabulario de la situación.
        '''
        tipos = list(self.entidades.keys())
        lista_aux = [self.entidades[l] for l in tipos]
        lista_aux = [item for sublist in lista_aux for item in sublist]
        self.vocabulario = [str(x) for x in lista_aux]
        self.vocabulario += [p.nombre for p in self.predicados]
        m = LogUtils.maxima_aridad(self.predicados) + 1
        lens = [len(self.vocabulario)]*m
        self.descriptor = Descriptor(lens)

    def fundamentar(self, expresion:nltk.sem.logic) -> nltk.sem.logic:
        '''
        Toma una fórmula en lpo de nltk y cambia los cuantificadores
        existenciales por Otorias y los cuantificadores universales
        por Ytorias. En ambos casos se utilizan las entidades y 
        eventos de la situación.
        Input:
            - expresión, que es un objeto fórmula en lpo de nltk
        Output:
            - fórmula fundamentada, que es un objeto fórmula en lpo de nltk
        '''
        tipo = LogUtils.obtener_type(expresion)
        if tipo in ['ExistsExpression']:
            # La expresión es un cuantificador existencial 
            # de una fórmula phi.
            # Determinamos si la variable del cuantificador es
            # o bien una entidad o bien un evento. 
            phi = expresion.term
            var = expresion.variable.name
            tipo_var = 'evento' if var[0] == 'e' else 'entidad'
            otoria = []
            if tipo_var == 'evento':
                consts = [str(c) for c in self.entidades['evento']]
                otoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            else:
                consts = [str(c) for c in self.entidades['individuo']]
                otoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            return LogUtils.Otoria([self.fundamentar(f) for f in otoria])
        elif tipo in ['AllExpression']:
            # La expresión es un cuantificador universal
            # de una fórmula phi.
            # Determinamos si la variable del cuantificador es
            # o bien una entidad o bien un evento. 
            phi = expresion.term
            var = expresion.variable.name
            tipo_var = 'evento' if var[0] == 'e' else 'entidad'
            if tipo_var == 'evento':
                consts = [str(c) for c in self.entidades['evento']]
#                print('\nYtoria sobre los eventos', consts)
                ytoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            else:
                consts = [str(c) for c in self.entidades['individuo']]
#                print('\nYtoria sobre las entidades', consts)
                ytoria = [self.nltk_log_parser.parse(rf'\{var}.({phi})({c})').simplify() for c in consts]
            return LogUtils.Ytoria([self.fundamentar(f) for f in ytoria])
        elif tipo in ['AndExpression']:
            first = self.fundamentar(expresion.first)
            second = self.fundamentar(expresion.second)
            return nltk.sem.logic.AndExpression(first, second)
        elif tipo in ['OrExpression']:
            first = self.fundamentar(expresion.first)
            second = self.fundamentar(expresion.second)
            return nltk.sem.logic.OrExpression(first, second)
        elif tipo in ['ImpExpression']:
            first = self.fundamentar(expresion.first)
            second = self.fundamentar(expresion.second)
            return nltk.sem.logic.ImpExpression(first, second)
        elif tipo in ['NegatedExpression']:
            term = self.fundamentar(expresion.term)
            return nltk.sem.logic.NegatedExpression(term)
        elif tipo in ['ApplicationExpression']:
            argumentos = expresion.args
            for x in argumentos:
                tipo_argumento = LogUtils.obtener_type(x)
                #assert('Constant' in tipo_argumento), f'¡Error: Átomo no fundamentado! {tipo_argumento} en {expresion}'
            return expresion
        elif tipo in ['EqualityExpression']:
            #assert(len(expresion.variables()) == 0), f'¡Error: Átomo no fundamentado! {expresion.variables()} en {expresion}'
            return expresion
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

    def codificar_lp(self, expresion:nltk.sem.logic) -> str:
        '''
        Toma una fórmula y devuelve su versión codificada 
        en lógica proposicional.
        Input:
            - expresión, que es un objeto fórmula en lpo de nltk
        Output:
            - codigo, que es un string en codificación lp
        '''
        tipo = LogUtils.obtener_type(expresion)
        if tipo in ['ExistsExpression', 'AllExpression']:
            raise Exception(f'¡Expresión no está fundamentada!')
        elif tipo in ['AndExpression']:
            first = self.codificar_lp(expresion.first)
            second = self.codificar_lp(expresion.second)
            return f'({first}∧{second})'
        elif tipo in ['OrExpression']:
            first = self.codificar_lp(expresion.first)
            second = self.codificar_lp(expresion.second)
            return f'({first}∨{second})'
        elif tipo in ['ImpExpression']:
            first = self.codificar_lp(expresion.first)
            second = self.codificar_lp(expresion.second)
            return f'({first}>{second})'
        elif tipo in ['NegatedExpression']:
            term = self.codificar_lp(expresion.term)
            return f'-{term}'
        elif tipo in ['ApplicationExpression', 'EqualityExpression']:
            return self.codificar_(expresion)
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

    def codificar_(self, pred:nltk.sem.logic.ApplicationExpression) -> str:
        '''
        Toma un predicado y devuelve su codificación
        Input:
            - pred, que es un ApplicationExpression de nltk
        Output:
            - codigo, que es un string
        '''
        tipo = LogUtils.obtener_type(pred)
        assert(tipo in ['ApplicationExpression', 'EqualityExpression'])
        if tipo == 'EqualityExpression':
            predicado = [self.vocabulario.index('IGUALDAD')]
            argumentos = [self.vocabulario.index(str(x)) for x in pred.constants()]
        else:
            try:
                predicado = [self.vocabulario.index(str(pred.pred))]
                argumentos = [self.vocabulario.index(str(x)) for x in pred.args]
            except Exception as e:
                raise Exception(f'¡Error al codificar formula atómica {pred}!\n{e}\n{self.vocabulario}')
        # print('')
        # print('-'*50)
        # print(f'vocabulario => {self.vocabulario}')
        # print(f'Codificando {pred}')
        # print(f'predicado {str(pred.pred)} index {self.vocabulario.index(str(pred.pred))}')
        # print(f'argumentos {pred.args} index {argumentos}')
        lista_valores = predicado + argumentos
        letra = self.descriptor.codifica(lista_valores=lista_valores)
        # print(f'Codificación: {letra}')
        return letra
    
    def decodificar(self, literal:str) -> str:
        assert(len(literal) <= 2), f'Literal incorrecto (se recibió {literal})'
        neg, atomo = PPT.como_literal(literal)
        lista_valores = self.descriptor.decodifica(atomo)
        predicado = self.vocabulario[lista_valores[0]]
        argumentos = [self.vocabulario[idx] for idx in lista_valores[1:]]
        argumentos = ', '.join(argumentos)
        formula_atomica = f'{neg}{predicado}({argumentos})'
        return formula_atomica

    def __str__(self):
        cadena = '\n' + '='*20 + 'COMPONENTES DEL MODELO' + '='*20
        cadena += '\n\nEntidades:\n'
        for tipo in self.entidades:
            cadena += f'\n\tTipo: {tipo}\n'
            for o in self.entidades[tipo]:
                cadena += f'\t=> {o}\n'
        cadena += '\nPredicados:\n\n'
        for p in self.predicados:
            cadena += f'\t=> {p.nombre}\n'
        cadena += '\n'
        cadena += '='*24 + 'FIN DEL MODELO' + '='*24 + '\n'
        return cadena


class Descriptor:
    '''
    Codifica una lista de N argumentos mediante un solo caracter
    '''

    def __init__ (self, args_lista, chrInit=256) :
        '''
        Input:  
            - args_lista, lista con el total de opciones para cada
                        argumento del descriptor
            - chrInit, entero que determina el comienzo de la codificación chr()
        Output: str de longitud 1
        '''
        self.args_lista = args_lista
        #assert(len(args_lista) > 0), "Debe haber por lo menos un argumento"
        self.chrInit = chrInit
        self.rango = [chrInit, chrInit + np.prod(self.args_lista)]

    def check_lista_valores(self,lista_valores) :
        for i, v in enumerate(lista_valores) :
            assert(v >= 0), "Valores deben ser no negativos"
            assert(v < self.args_lista[i]), f"Valor debe ser menor o igual a {self.args_lista[i]}"

    def lista_a_numero(self,lista_valores) :
        self.check_lista_valores(lista_valores)
        cod = lista_valores[0]
        n_columnas = 1
        for i in range(0, len(lista_valores) - 1) :
            n_columnas = n_columnas * self.args_lista[i]
            cod = n_columnas * lista_valores[i+1] + cod
        return cod

    def numero_a_lista(self,n) :
        decods = []
        if len(self.args_lista) > 1:
            for i in range(0, len(self.args_lista) - 1) :
                n_columnas = np.prod(self.args_lista[:-(i+1)])
                decods.insert(0, int(n / n_columnas))
                n = n % n_columnas
        decods.insert(0, n % self.args_lista[0])
        return decods

    def codifica(self,lista_valores) :
        codigo = self.lista_a_numero(lista_valores)
        return chr(self.chrInit+codigo)

    def decodifica(self,codigo) :
        n = ord(codigo)-self.chrInit
        return self.numero_a_lista(n)
    

class PPT :

	'''
	Pre-processing tools
	'''
    
	@staticmethod
	def Ytoria_LaTeX(
				lista_forms: List[str], 
				left_first: Optional[bool]=True
			) -> str:
		form = ''
		inicial = True
		if left_first:
			lista = lista_forms[::-1]
			backward=False
		else:
			lista = lista_forms
			backward=True
		for f in lista:
			if inicial:
				form = f
				inicial = False
			else:
				if backward:
					form = '(' + form + r'\wedge' + f + ')'
				else:
					form = '(' + f + r'\wedge' + form + ')'
		return form
	
	@staticmethod
	def clausal_a_LaTeX(S: List[List[str]]) -> str:
		
		def preprocess(C: List[str]):
			new_C = list()
			for c in C:
				neg, l = PPT.como_literal(c)
				if 'x' in l:
					c = neg + r'x_{' + c[1:] + '}'
				if '-' in c:
					c = c.replace('-', r'\neg ')
				new_C.append(c)
			return new_C
		
		aux = ['[' + ', '.join(preprocess(C)) + ']' for C in S]
		aux = '[' + ', '.join(aux) + ']'
		return str(aux)

	@staticmethod	
	def como_literal(l:str) -> Tuple[str, str]:
		if '-' in l:
			return '-', l[1:]
		else:
			return '', l

	@staticmethod
	def mostrar_como_atomos_numerados(
			S: List[str], 
			letrasp: List[str],
		) -> List[str]:
		'''
		Algoritmo para visualizar una lista de literales como una lista de átomos de la forma x_n
		Input: 
			- S, una lista de literales,
			- letrasp, una lista de letras proposicionales
		Output: 
			- S_, una lista de literales 
		'''
		nueva_c = list()
		for c in S:
			neg, l = PPT.como_literal(c)
			assert(l in letrasp), f'Error: el caracter {l} debe estar en\n\t{letrasp}'
			indice = letrasp.index(l)
			nueva_c.append(f'{neg}x{indice + 1}')
		return nueva_c

	@staticmethod
	def mostrar_como_formula_fundamentada(
			literal: str,
			letrasp: List[str],
			len_tseitin: int,
		) -> str:
		'''
		Convierte el literal de la forma x_n o -x_n usando la letra proposicional de indice n
		'''
		neg, atomo = PPT.como_literal(literal)
		assert(atomo[0] == 'x'), f'Error: literal incorrecto {literal}. Debe ser de la forma x_n.'
		indice = int(atomo[1:])
		if indice < len(letrasp):
			return neg + letrasp[indice]
		else:	
			return f'{neg}x{indice - len_tseitin}'
		
	@staticmethod
	def mostrar_como_entero(literal):
		neg, atomo = PPT.como_literal(literal)
		if neg == '-':
			return -1 * int(atomo[1:])
		else:
			return int(atomo[1:])