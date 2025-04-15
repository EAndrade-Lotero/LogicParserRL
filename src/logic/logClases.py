'''
Objetos del lenguaje de lógica de primer orden
que son la base de una situación.
'''
class Constante:
    '''
    Define las constantes del lenguaje. Estas representarán
    los individuos del universo.
    '''
    def __init__(self, tipo:str='individuo', nombre:str=''):
        self.tipo = tipo
        self.nombre = nombre

    def en_conjunto(self, constantes:list) -> bool:
        '''
        Determina si la constante está en un conjunto
        de constantes dado.
        Input:
            - constantes, lista de constantes
        Output:
            - True/False
        '''
        nombres = [c.nombre for c in constantes]
        return self.nombre in nombres

    def __str__(self):
        return self.nombre

class Predicado:
    '''
    Define las constantes de predicado del lenguaje.
    Requiere un nombre y una lista de tipos (como cadenas).
    '''
    def __init__(self, nombre:str, tipos_argumentos:list):
        self.nombre = nombre
        self.aridad = len(tipos_argumentos)
        self.tipos_argumentos = tipos_argumentos

    def en_conjunto(self, predicados:list) -> bool:
        '''
        Determina si la constante está en un conjunto
        de predicados dado.
        Input:
            - predicados, lista de predicados
        Output:
            - True/False
        '''
        nombres = [p.nombre for p in predicados]
        return self.nombre in nombres

    def __str__(self):
        return self.nombre