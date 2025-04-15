from tqdm import tqdm
from copy import deepcopy
from typing import List, Tuple

from logic.codificacion import PPT

class TseitinTransform :

    '''
    Implements the Tseitin transformation of a formula
    '''

    def __init__(self) -> None:
        self.conectivos_binarios = ['∧','∨','>','=']
        self.latex_conectivos = [r'\wedge', r'\vee', r'\to', r'\leftrightarrow']
        self.atomos = list()
        self.atomos_tseitin = list()
        self.debug = False

    def a_clausal(self, A:str) -> List[str]:
        # Subrutina de Tseitin para encontrar la FNC de
        # la formula en la pila
        # Input: A (cadena) de la forma
        #                   p=-q
        #                   p=(q∧r)
        #                   p=(q∨r)
        #                   p=(q>r)
        # Output: B (cadena), equivalente en FNC
        assert(len(A)==4 or len(A)==7), u"Fórmula incorrecta!"
        B = ''
        p = A[0]
        # print('p', p)
        if "-" in A:
            q = A[-1]
            if self.debug:
                print('q', q)
            B = "-"+p+"∨-"+q+"∧"+p+"∨"+q
        elif "∧" in A:
            q = A[3]
            if self.debug:
                print('q', q)
            r = A[5]
            if self.debug:
                print('r', r)
            B = q+"∨-"+p+"∧"+r+"∨-"+p+"∧-"+q+"∨-"+r+"∨"+p
        elif "∨" in A:
            q = A[3]
            if self.debug:
                print('q', q)
            r = A[5]
            if self.debug:
                print('r', r)
            B = "-"+q+"∨"+p+"∧-"+r+"∨"+p+"∧"+q+"∨"+r+"∨-"+p
        elif ">" in A:
            q = A[3]
            if self.debug:
                print('q', q)
            r = A[5]
            if self.debug:
                print('r', r)
            B = q+"∨"+p+"∧-"+r+"∨"+p+"∧-"+q+"∨"+r+"∨-"+p
        elif "=" in A:
            q = A[3]
            if self.debug:
                print('q', q)
            r = A[5]
            if self.debug:
                print('r', r)
            B = q+"∨"+"-"+r+"∨"+"-"+p+"∧"+"-"+q+"∨"+r+"∨"+"-"+p+"∧"+"-"+q+"∨"+"-"+r+"∨"+p+"∧"+q+"∨"+r+"∨"+p
        else:
            raise Exception(f'Error enENC(): Fórmula incorrecta! ({A})')
        B = B.split('∧')
        B = [c.split('∨') for c in B]
        return B

    def tseitin(self, A:str) -> List[List[str]]:
        '''
        Algoritmo de transformacion de Tseitin
        Input: A (cadena) en notacion inorder
        Output: B (cadena), Tseitin
        '''
        # Creamos letras proposicionales nuevas
        num_conectivos_binarios = sum([A.count(c) for c in self.conectivos_binarios])
        if self.debug:
            print('Número de conectivos binarios:', num_conectivos_binarios)
        num_negaciones = A.count('-')
        if self.debug:
            print('Número de negaciones:', num_negaciones)
        simbolos_especiales = ['(', ')', '-'] + self.conectivos_binarios
        aux = deepcopy(A)
        for c in simbolos_especiales:
            aux = aux.replace(c, '')
        letrasp = list(set(list(aux)))
        self.atomos = letrasp
        if self.debug:
            print('Letras proposicionales en A:', letrasp)
            print('Número de letras proposicionales:', len(letrasp))
        cods_letras = [ord(x) for x in letrasp]
        m = max(cods_letras) + 256
        letrasp_tseitin = [chr(x) for x in range(m, m + num_conectivos_binarios + num_negaciones)]
        self.atomos_tseitin = letrasp_tseitin
        if self.debug:
            print('Número de letras Tseitin:', len(letrasp_tseitin))
        letrasp = list(letrasp) + letrasp_tseitin
        if self.debug:
            for i, l in enumerate(letrasp):
                print(f'\t{l} => x{i + 1}')
        L = [] # Inicializamos lista de conjunciones
        Pila = [] # Inicializamos pila
        i = -1 # Inicializamos contador de variables nuevas
        s = A[0] # Inicializamos símbolo de trabajo
        if self.debug:
            pbar = tqdm(total=len(A))
        else:
            pbar = tqdm(total=len(A), disable=True)
        while len(A) > 0: # Recorremos la cadena
            if self.debug:
                print(A)
                print("Pila:", Pila, " L:", L, " s:", s)
            if (s in letrasp) and (len(Pila) > 0) and (Pila[-1]=='-'):
                i += 1
                atomo = letrasp_tseitin[i]
                Pila = Pila[:-1]
                Pila.append(atomo)
                L.append(atomo + "=-" + s)
                if self.debug:
                    print(f'A la pila: {atomo + "=-" + s}')
                    pbar.update(1)
                A = A[1:]
                if len(A) > 0:
                    s = A[0]
            elif s == ')':
                left = Pila[-3]
                conectivo = Pila[-2]
                assert conectivo in self.conectivos_binarios, u"Error en la pila!"
                right = Pila[-1]
                Pila = Pila[:len(Pila)-4]
                i += 1
                atomo = letrasp_tseitin[i]
                L.append(atomo + "=(" + left + conectivo + right + ")")
                if self.debug:
                    print(f'A la pila: {atomo + "=(" + left + conectivo + right + ")"}')
                s = atomo
            else:
                Pila.append(s)
                A = A[1:]
                if self.debug:
                    pbar.update(1)
                if len(A) > 0:
                    s = A[0]
        if i < 0:
            atomo = Pila[-1]
        else:
            atomo = letrasp_tseitin[i]
        B = [[[atomo]]] + [self.a_clausal(x) for x in L]
        B = [val for sublist in B for val in sublist]
        return B





