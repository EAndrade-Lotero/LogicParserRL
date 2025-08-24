import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from parser.clases_parser import ParserFOL_1f
from parser.PER_SB3 import PERDQN
from config.config import PATHS


def train_test_split(n: int):
    df = pd.read_csv(PATHS['fol_data_folder'] / 'smallest_ordered_replaced.csv')
    df.columns = ["frase","frase-FOL","pregunta","respuesta","tipo_pregunta"]
    df = df.sort_values(by="frase", key=lambda col: col.str.len())
    df.reset_index(drop=True, inplace=True)

    df_train = df.iloc[1:n + 1, :]
    df_test = df.iloc[n + 1:, :]

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test


print("Running training from scratch...")

# Dataset reducido
n_samples = 50
df_train_n, df_test_n = train_test_split(n_samples)

# Crear entorno con el dataset de entrenamiento
env = ParserFOL_1f(df_train_n)
env.max_turns = 7
env.pick_first = True
env.prob_select_first = True

# Crear el agente desde cero
model = PERDQN(
    "MlpPolicy",   # tipo de red
    env,
    learning_rate=1e-3,
    batch_size=32,
    buffer_size=50000,
    learning_starts=1000,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    verbose=1
)

# Entrenamiento
timesteps = 1000  # 100k 
model.learn(total_timesteps=timesteps)

# Guardar modelo entrenado
model_name = "PER_ALL_SOME_v5.2_scratch"
output_path = PATHS['models'] / f"model_{model_name}"
model.save(output_path)

print("Entrenamiento finalizado y modelo guardado.")
