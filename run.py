import retro
import numpy as np
import torch
import cv2
from mario_dqn import MarioDQN

def calculate_reward(info, next_state, done):
    reward = 0

    if 'x' in info:
        reward = reward + 1
    
    if done:
        reward = reward - 10
    
    return reward

def preprocess_state(state):
    # converter para tons de cinza
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    # redimesiona para 84x84
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_AREA)
    # normaliza os valores (opcional, pode ser útil para treinamento)
    state = state / 255.0
    state = torch.FloatTensor(state).unsqueeze(0) # Adiciona canal [1,84,84]
    return state

# Configura o ambiente e o modelo
env = retro.make(game='SuperMarioWorld-Snes')
mario_agent = MarioDQN(action_dim=env.action_space.n)

num_epsodes = 500 # Numero total de epsodios
max_steps = 1000 # Máximo de passos por epsodio

for episode in range(num_epsodes):
    state = env.reset()
    state = preprocess_state(state)
    state = state.unsqueeze(0) # Adiciona dimensão do batch: [1,1,84,84]
    episode_reward = 0 # Recompensa acumulada no epsodio

    if episode % 10 == 0:  # Salvar a cada 10 episódios
        torch.save(mario_agent.network.state_dict(), f"mario_dqn_episode_{episode}.pth")

    for step in range(max_steps):
        action = mario_agent.choose_action(state)

        # Executa a ação e observa o proximo passo e recompensa
        next_state, reward, done, info = env.step(action)
        next_state = preprocess_state(next_state)
        next_state = next_state.unsqueeze(0)

        for key, value in info.items():
            print(f'{key}: {value}')

        # Calcula a recompensa e armazena a experiencia
        reward = calculate_reward(info, next_state, done)
        mario_agent.store_experience(state, action, reward, next_state, done)

        # Atualiza o estado atual para o proximo
        state = next_state
        episode_reward = episode_reward + reward

        # Realiza o treinamento da rede com as experiencias armazenadas
        mario_agent.train(batch_size=64)

        # Exibe o ambiente para visualização
        env.render()

        if done:
            break
    
    # Impriome a recompensa acumulada do episodio
    print(f"Episodio {episode + 1}/{num_epsodes} - Recompensa: {episode_reward}")

env.close()