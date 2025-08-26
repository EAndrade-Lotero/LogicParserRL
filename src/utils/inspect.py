import numpy as np

from copy import deepcopy
from prettytable import PrettyTable


def q_values_at_state(model, env, obs):
    state_tensor, _ = model.policy.obs_to_tensor(np.array(obs))
    model.q_net.eval()
    values = model.policy.q_net.forward(state_tensor).squeeze()
    q_values = zip(env.nombre_acciones, values.tolist())
    q_values = sorted(q_values, key=lambda x: x[1], reverse=True)

    table = PrettyTable()
    table.field_names = ["Acción", "Valor"]

    for action, value in q_values:
        table.add_row([action, value])

    print(table)

def rewards_at_state(env, list_prev_actions):

    env_ = deepcopy(env)
    rewards = []
    for action, action_name in enumerate(env_.nombre_acciones):
        env_.reset()
        for a in list_prev_actions:
            env_.step(a)
        obs, reward, terminated, truncated, info = env_.step(action)
        rewards.append(reward)

    reward_values = zip(env_.nombre_acciones, rewards)
    reward_values = sorted(reward_values, key=lambda x: x[1], reverse=True)

    table = PrettyTable()
    table.field_names = ["Acción", "Recompensa"]

    for action, value in reward_values:
        table.add_row([action, value])

    print (table)


def quick_inspection(model, env, acciones):

    env_ = deepcopy(env)
    obs, info = env_.reset()
    counter = 0
    total_reward = 0
    inicial = True
    list_states = [obs]
    list_actions = []

    while True:
        counter += 1
        print('')
        print('='*60)
        print(f"Iteración {counter}")
        print('')
        print('-'*60)
        state = {"Estado": env_.estado, "Raiz": env_.raiz}
        estado_actual, raiz = state['Estado'], state['Raiz'] 
        state_str = f"Índice: {estado_actual.indice}\n"
        state_str += f"Nivel: {estado_actual.nodo.nivel()}\n"
        state_str += f"{estado_actual.frase}\n"
        state_str += f"{estado_actual.obtener_cadena()}\n"
        msg = str(raiz.simplificar2recompensa().fol())
        state_str +=  f"{msg}"
        print(state_str)
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
        else:
            print(f'Tamaño diferencia: {len(np.where(obs != obs_anterior)[0])}')
        total_reward += reward
        print('-'*60)
        print('')
        print(f"accion: {env_.nombre_acciones[action]}")
        print(f'Recompensa: {reward}')
        # print(f'Estoy en el nivel {env.estado.nodo.nivel()}')
        if terminated or truncated:
            print('')
            print('='*60)
            state = {"Estado": env_.estado, "Raiz": env_.raiz}
            estado_actual, raiz = state['Estado'], state['Raiz'] 
            state_str = f"Índice: {estado_actual.indice}\n"
            state_str += f"Nivel: {estado_actual.nodo.nivel()}\n"
            state_str += f"Frase inicial: {estado_actual.frase}\n"
            state_str += f"Frase procesada: {estado_actual.obtener_cadena()}\n"
            msg = str(raiz.simplificar2recompensa().fol())
            state_str +=  f"FOL: {msg}"
            print(state_str)
            print(f'terminated: {terminated}, truncated: {truncated}')
            break

    print(f'Total reward: {total_reward}')

    return list_states, list_actions