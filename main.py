import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

#import sys
#sys.path.append('../src')

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


from parser.clases_parser import ParserFOL_1f
from parser.PER_SB3 import PERDQN, PrioritizedReplayBuffer
from config.config import PATHS

def train_test_split(n:int):
    #df = pd.read_csv(PATHS['fol_data_folder']/'smallest_ordered_replaced.csv')
    df = pd.read_csv(PATHS['fol_data_folder'] / 'equivalencia_5_frase_fol.csv')
    df.columns = ["frase","frase-FOL","pregunta","respuesta","tipo_pregunta"]
    df = df.sort_values(by="frase", key=lambda col: col.str.len())
    df.reset_index(drop=True, inplace=True)

    df_train = df.iloc[1:n+1, :]
    df_test = df.iloc[n+1:, :]

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    return df_train, df_test

def evaluation(model, env, acciones):

    env_ = env # deepcopy(env)
    obs, info = env_.reset()
    counter = 0
    total_reward = 0
    inicial = True
    list_states = [obs]
    list_actions = []

    while True:
        counter += 1
        if counter - 1 >= len(acciones):
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = env_.accion_to_index(acciones[counter - 1])
        list_actions.append(action)
        obs_anterior = obs
        obs, reward, terminated, truncated, info = env_.step(action)
        list_states.append(obs)
        if inicial:
            inicial = False
        total_reward += reward
        # print(f'Estoy en el nivel {env.estado.nodo.nivel()}')
        if terminated or truncated:
            break
    #print(f"Frase del entorno: {env_.frase} ---- Recompensa {total_reward} ---- Done {terminated}")
    return total_reward, terminated, env_.get_formula_actual()


print('Running...')

n_samples = 5
df_train_n, df_test_n = train_test_split(n_samples)

# Instanciar el agente
#df = pd.read_csv(PATHS['fol_data_folder']/'smallest_ordered_replaced.csv')
df = pd.read_csv(PATHS['fol_data_folder'] / 'equivalencia_5_frase_fol.csv')
env = ParserFOL_1f(df)
model_name = "PER_ALL_SOME_v5.2_scratch"
model_path = PATHS['models'] / f"model_{model_name}"
model = PERDQN.load(model_path, env=env)

acciones = []
rewards_some = []
dones_some = []
frases_some = []
rewards_all = []
dones_all = []
frases_all = []

# data = df_train_n
data = df_test_n

iteraciones = len(data)
for num_frase in range(iteraciones):

    # Filtrar por frase
    df = pd.DataFrame(data.loc[num_frase].to_frame().T)
    df.reset_index(drop=True, inplace=True)

    # Crear el entorno con la frase inicial   
    env = ParserFOL_1f(df) #_simply
    env.max_turns = 7
    env.pick_first = True
    env.prob_select_first = True
    env.reset()

    # Evaluar frase inicial
    r_some, d_some, frase_some = evaluation(model, env, acciones)
    print(f'Frase: {env.frase} ---  FOL obtenido: {frase_some}')
    rewards_some.append(r_some)
    dones_some.append(d_some)
    frases_some.append(frase_some)

    # Crear el entorno con la frase final   
    env = ParserFOL_1f(df) #_simply
    env.max_turns = 7
    env.pick_first = False
    env.prob_select_first = False
    env.reset()

    # Evaluar frase final
    r_all, d_all, frase_all = evaluation(model, env, acciones)
    print(f'Frase: {env.frase} ---  FOL obtenido: {frase_all}')
    rewards_all.append(r_all)
    dones_all.append(d_all)
    frases_all.append(frase_all)

print("Metric Some:", sum(dones_some)/iteraciones)
print("Metric All:", sum(dones_all)/iteraciones)