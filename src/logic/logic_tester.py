import pycosat
from typing import List

from logic.logUtils import LogUtils
from logic.tseitin import TseitinTransform
from logic.codificacion import ToPropositionalLogic, ToNumeric


class LogicTester:

    def __init__(self):
        self.to_lp = ToPropositionalLogic()
        # self.to_lp.debug = True
        self.tseitin = TseitinTransform()
        # self.tseitin.debug = True
        self.debug = False
    
    def negate_sentence(self, sentence:str) -> str:
        '''
        Negate a sentence.
        '''
        if sentence[0] != '-':
            negated_sentence = f'-{sentence}'
        else:
            negated_sentence = sentence[1:]
        return negated_sentence

    def translation_to_prover(self,sentence:str) -> str:
        '''
        Translate a sentence to prover format.
        '''
        sentence_lp = self.to_lp.parse(sentence)
        return sentence_lp

    def check_implication(self, premisas:List[any], conclusion:any) -> bool:
        if len(premisas) == 0:
            formula = conclusion
        elif len(premisas) == 1:
    #        formula = f'-({premisas[0]}>{conclusion})'
            formula = f'({premisas[0]}∧{self.negate_sentence(conclusion)})'
        elif len(premisas) > 1:
            premisas_ = LogUtils.Ytoria(premisas)
    #        formula = f'-({premisas_}>{conclusion})'
            formula = f'({premisas_}∧{self.negate_sentence(conclusion)})'
        formula_tseitin = self.tseitin.tseitin(formula)
        to_numeric = ToNumeric(formula_tseitin)
        formula_numeros = to_numeric.to_numeric(formula_tseitin)
        res = pycosat.solve(formula_numeros)
        if self.debug:
            print('Las premisas son:\n')
            for p in premisas:
                print('\t', p, end='\n\n')
            print('\nLa conclusion es:\n\n\t', conclusion)
            print(f'La fórmula a chequear es:\n\n\t{formula}')
            if res == 'UNSAT':
                print('\n¡La conclusión se sigue lógicamente de las premisas!')
            else:
                print('\n¡La conclusión NO se sigue lógicamente de las premisas')
                modelo = [to_numeric.literal(x) for x in res]
                modelo = [x for x in modelo if to_numeric.solo_atomo(x) in self.tseitin.atomos] 
                print(f'\nUn modelo es:\n\n\t{modelo}')
        return (res == 'UNSAT')

    def test_negacion(self, sentence1:str, sentence2:str) -> bool:
        '''
        Test negation between two sentences.
        '''
        #Test sentence1 implies -sentence2
        sentence1_prover = self.translation_to_prover(sentence1)
        negated_sentence2_prover = self.translation_to_prover(self.negate_sentence(sentence2))
        premisas = [sentence1_prover]
        conclusion = negated_sentence2_prover
        resultado1 = self.check_implication(premisas, conclusion)
        # print('Resultado:', resultado1)

        # Test -sentence1 implies sentence2
        negated_sentence1_prover = self.translation_to_prover(self.negate_sentence(sentence1))
        sentence2_prover = self.translation_to_prover(sentence2)
        premisas = [negated_sentence1_prover]
        conclusion = sentence2_prover
        resultado2 = self.check_implication(premisas, conclusion)
        # print('Resultado:', resultado2)

        return resultado1 and resultado2