'''
Funciones auxiliares sobre fórmulas en lógica de primer orden (lpo).
'''
import nltk
from logic.logClases import *

lp = nltk.sem.logic.LogicParser()

class LogUtils:
    '''
    Herramientas para trabajar con fórmulas de lógica de primer orden (lpo).
    '''
    def __init__(self):
        pass

    @staticmethod
    def obtener_type(objeto):
        '''Toma un objeto y retorna su tipo de manera legible
        Input:
            - objeto
        Output:
            - tipo, cadena con el tipo del objeto
        '''
        c = str(type(objeto))
        return c.split('.')[-1][:-2]
    
    @staticmethod
    def sust(var:nltk.sem.logic, exp1:nltk.sem.logic, exp2:nltk.sem.logic) -> nltk.sem.logic:
        '''
        Sustituye una expresión en otra.
        Input:
            - var, que es una variable en lpo de nltk
            - exp1, que es una fórmula en lpo de nltk
            - exp2, que es una fórmula en lpo de nltk
        Output:
            - expresion, en la cual exp2 se ha sustituido por las
                        ocurrencias libres de var en exp1
        '''
        aux = lp.parse(rf'\{var}.({exp1})({exp2})')
        return aux.simplify()
    
    @staticmethod
    def desempaquetar(lista:list) -> str:
        '''
        Toma una lista de formulas y devuelve un solo string unido por comas
        '''
        nombres = [str(x) for x in lista]
        unir = ','.join(nombres)
        return unir
    
    @staticmethod
    def sust_vars(vars:list, exp1:nltk.sem.logic, exp2:list) -> nltk.sem.logic:
        '''
        Sustituye una lista de expresiones en otra.
        Input:
            - vars, que es una lista de variables en lpo de nltk
            - exp1, que es una fórmula en lpo de nltk
            - exp2, que es una lista de fórmulas en lpo de nltk
        Output:
            - expresion, en la cual las exp2 se han sustituido por las
                        ocurrencias libres de vars en exp1
        '''
        aux = lp.parse(rf'\{LogUtils.desempaquetar(vars)}.({exp1})({LogUtils.desempaquetar(exp2)})')
        return aux.simplify()

    @staticmethod
    def remover_existencial(expresion:nltk.sem.logic, constante:nltk.sem.logic) -> nltk.sem.logic:
        '''
        Toma una fórmula de tipo existe x phi(x), en la
        cual se sustituyen todas las ocurrencias libres de x 
        por una constante.
        Input:
            - expresion, que es una fórmula en lpo de nltk
            - constante, que es una constante en lpo de nltk
        Output:
            - formula, que es un objeto lpo de nltk donde la variable
                    del existencial fue reemplazada por la constate.
        '''
        tipo = LogUtils.obtener_type(expresion)
        assert(tipo == 'ExistsExpression'), f'¡La fórmula debe ser de tipo existencial!\nSe obtuvo {tipo}'
        var = expresion.variable
        funcion = expresion.term
        return LogUtils.sust(var=var, exp1=funcion, exp2=constante)

    @staticmethod
    def existenciales_a_constantes(expresion:nltk.sem.logic) -> nltk.sem.logic:
        '''
        Toma una fórmula que tiene existenciales y los cambia por 
        una constante del tipo respectivo (individuo o evento).
        '''
        constantes, predicados = LogUtils.obtener_vocabulario(expresion)
        tipo = LogUtils.obtener_type(expresion)
        if tipo in ['ExistsExpression']:
            var = expresion.variable.name
            nombre_ = LogUtils.encuentra_nombre(variable=expresion.variable, predicados=predicados, formula=str(expresion.term))
            if nombre_ is not None:
                if var[0] == 'e':
                    nombre = 'Ev_' + nombre_
                else:
                    nombre = nombre_
                constante = lp.parse(nombre)
                formula = LogUtils.remover_existencial(expresion=expresion, constante=constante)
                return LogUtils.existenciales_a_constantes(formula)
        elif tipo in ['AllExpression']:
            term = LogUtils.existenciales_a_constantes(expresion.term)
            variable  = list(term.free())[0]
            return nltk.sem.logic.AllExpression(variable, term)
        elif tipo in ['NegatedExpression']:
            term = LogUtils.existenciales_a_constantes(expresion.term)
            return nltk.sem.logic.NegatedExpression(term)
        elif tipo in ['AndExpression']:
            first = LogUtils.existenciales_a_constantes(expresion.first)
            second = LogUtils.existenciales_a_constantes(expresion.second)
            return nltk.sem.logic.AndExpression(first, second)
        elif tipo in ['OrExpression']:
            first = LogUtils.existenciales_a_constantes(expresion.first)
            second = LogUtils.existenciales_a_constantes(expresion.second)
            return nltk.sem.logic.OrExpression(first, second)
        elif tipo in ['ImpExpression']:
            first = LogUtils.existenciales_a_constantes(expresion.first)
            second = LogUtils.existenciales_a_constantes(expresion.second)
            return nltk.sem.logic.ImpExpression(first, second)
        elif tipo in ['ApplicationExpression', 'EqualityExpression']:
            return expresion
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

    @staticmethod
    def unir_constantes(consts1:list, consts2:list) -> list:
        '''
        Toma dos listas de constantes y devuelve una sola
        sin repeticiones.
        Input:
            -consts1, lista de objetos constante
            -consts2, lista de objetos constante
        Output
            - lista de objetos constante
        '''
        unicos = [c for c in consts1]
        for c in consts2:
            if not c.en_conjunto(consts1):
                unicos.append(c)
        return unicos
 
    @staticmethod
    def unir_predicados(preds1:list, preds2:list) -> list:
        '''
        Toma dos listas de predicados y devuelve una sola
        sin repeticiones.
        Input:
            -consts1, lista de objetos predicado
            -consts2, lista de objetos predicado
        Output
            - lista de objetos predicado
        '''
        unicos = [p for p in preds1]
        for p in preds2:
            if not p.en_conjunto(preds1):
                unicos.append(p)
        return unicos

    @staticmethod
    def obtener_vocabulario(expresion:nltk.sem.logic) -> list:
        '''
        Toma una fórmula en lpo de nltk y devuelve sus
        constantes y predicados como objetos de parseSit.
        Input:
            - expresion, que es una fórmula en lpo de nltk
        Output:
            - constantes, que es un conjunto de 
                        constantes (como objetos de parseSit)
            - predicados, que es un conjunto de
                        predicados (como objetos de parseSit)
        '''
        tipo = LogUtils.obtener_type(expresion)
        #print(f'tipo: {tipo}')
        if tipo in ['ExistsExpression', 'AllExpression', 'LambdaExpression', 'NegatedExpression']:
            return LogUtils.obtener_vocabulario(expresion.term)
        elif tipo in ['AndExpression', 'OrExpression', 'ImpExpression']:
            constantes1, predicados1 = LogUtils.obtener_vocabulario(expresion.first)
            constantes2, predicados2 = LogUtils.obtener_vocabulario(expresion.second)
            return  LogUtils.unir_constantes(constantes1, constantes2), LogUtils.unir_predicados(predicados1, predicados2)
        elif tipo in ['ApplicationExpression']:
            # Creamos el predicado
            predicados_ = expresion.predicates()
            assert(len(predicados_) == 1)
            argumentos = expresion.args
            #print(argumentos)
            tipos_argumentos = [LogUtils.obtener_type(x) for x in argumentos]
            #print(tipos_argumentos)
            predicado = Predicado(
                nombre=str(list(predicados_)[0]), 
                tipos_argumentos=tipos_argumentos
            )
            #print(predicado.nombre)
            #print(predicado.tipos_argumentos)
            predicados = [predicado] 
            # Creamos las constantes
            constantes = []
            for x in argumentos:
                #print('for x in argumentos')
                #print(x)
                tipo_x = LogUtils.obtener_type(x)
                if 'Constant' in tipo_x:
                    if str(x)[0:3] == 'Ev_':
                        tipo_constante = 'evento'
                    else:
                        tipo_constante = 'individuo'
                    c = Constante(tipo=tipo_constante, nombre=str(x))
                    if not c.en_conjunto(constantes):
                        constantes.append(c)
            return  constantes, predicados
        elif tipo in ['EqualityExpression']:
            constantes = []
            for x in expresion.constants():
                if str(x)[0:3] == 'Ev_':
                    tipo_constante = 'evento'
                else:
                    tipo_constante = 'individuo'
                c = Constante(tipo=tipo_constante, nombre=str(x))
                if not c.en_conjunto(constantes):
                    constantes.append(c)    
            predicado = Predicado(nombre='IGUALDAD', tipos_argumentos=['any', 'any'])
            predicados = [predicado] 
            return constantes, predicados
        else:
            raise Exception(f'¡Tipo de expresión desconocido! {tipo}')

    @staticmethod
    def encuentra_nombre(variable:str, predicados:list, formula) -> str:
        '''
        Toma una variable y una lista de predicados y devuelve el nombre de la entidad.
        Por ejemplo, devuelve "perro" si variable es "x" y "PERRO(x)" está en la fórmula.
        Input:
            - variable, cadena con el nombre de la variable
            - predicados, lista de objetos Predicado
            - formula, cadena con la fórmula
        Output:
            - nombre, cadena
        '''
        nombre = None
        for p in predicados:
            inicial = True
            if p.aridad == 1:
                prueba = f'{p.nombre}({variable})'
                if prueba in formula:
                    if inicial:
                        nombre = p.nombre.lower()
                        inicial = False
                    else:
                        nombre = f'{nombre}-{p.nombre.lower()}'
        return nombre

    @staticmethod
    def Ytoria(lista_forms:list) -> nltk.sem.logic:
        '''
        Toma una lista de formulas y las une mediante &.
        Input:
            - lista_forms, que es una lista de fórmulas como objetos de nltk
        Output:
            - formula, que es un objeto de nltk
        '''
        if len(lista_forms) == 0:
            return None
        elif len(lista_forms) == 1:
            return lista_forms[0]
        else:
            form = lista_forms[0]
            for f in lista_forms[1:]:
                form = nltk.sem.logic.AndExpression(form, f)
            return form

    @staticmethod
    def Otoria(lista_forms:list) -> nltk.sem.logic:
        '''
        Toma una lista de formulas y las une mediante |.
        Input:
            - lista_forms, que es una lista de fórmulas como objetos de nltk
        Output:
            - formula, que es un objeto de nltk
        '''
        if len(lista_forms) == 0:
            return None
        elif len(lista_forms) == 1:
            return lista_forms[0]
        else:
            form = lista_forms[0]
            for f in lista_forms[1:]:
                form = nltk.sem.logic.OrExpression(form, f)
            return form
 
    @staticmethod
    def maxima_aridad(predicados:list) -> int:
        '''
        Toma una lista de predicados y devuelve la aridad máxima
        '''
        aridades = [p.aridad for p in predicados]
        return max(aridades)

    @staticmethod
    def diferenciar_eventos(formulas:list) -> list:
        '''
        Toma una lista de fórmulas y enumera los eventos
        Input:
            - formulas, lista de fórmulas en lógica de primer orden con un evento como variable libre
        Output:
            - diferenciadas, lista de fórmulas de lpo con eventos enumerados
        '''
        lp = nltk.sem.logic.LogicParser()
        diferenciadas = []
        for i, f in enumerate(formulas):
            print('formula',f)
            libres = f.variables()
            print('variables de la formula', libres)
            nombres = [v.name for v in libres]
            assert('e' in nombres), f'No se halló ningun evento en {nombres}'
            exp = lp.parse(rf'(\e.{f})(e{i})').simplify()
            exp1 = lp.parse(rf'exists e{i}.({exp})')
            diferenciadas.append(exp1)
        return diferenciadas
 
    @staticmethod
    def anaforicas(formula:nltk.sem.logic) -> list:
        '''
        Toma una fórmula de lpo y devuelve las variables libres de individuos
        Input:
            - formula, objeto nltk de lógica de primer orden
        Output:
            - libres, lista de variables anafóricas
        '''
        libres_ = formula.free()
        libres = [v.name for v in libres_ if v.name[0] != 'e']
        return libres
    
    @staticmethod
    def predicados_a_existenciales(expresion:nltk.sem.logic) -> nltk.sem.logic:
        '''
        Toma una fórmula y crea una formula con existenciales por cada predicado
        '''
        constantes, predicados = LogUtils.obtener_vocabulario(expresion)
        lista_existenciales = list()
        for p in predicados:
            if p.aridad == 1:
                formula_existencial = lp.parse(rf'exists x.{p.nombre}(x)')
                lista_existenciales.append(formula_existencial)
        if len(lista_existenciales) > 0:
            existenciales = LogUtils.Ytoria(lista_existenciales)
            return existenciales
        else:
            return None