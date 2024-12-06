import random
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import torch.nn.functional as F
from mario_net import MarioNet

class MarioDQN:
    def __init__(self, action_dim):
        self.network = MarioNet(action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        self.memory = deque(maxlen=10000) # Buffer de experiencia
        self.gamma = 0.99 # Fator de desconto
        self.epsilon = 1.0 # Taxa de exploração inicial
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.action_dim = action_dim
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample_experience(self, batch_size=64):
        return random.sample(self.memory, batch_size)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Retorna uma ação aleatoria no formato esperado (vetor de botões)
            action = np.random.choice([0,1], size=(self.action_dim,))
        else:
            # Ação baseada na rede
            if state.ndim == 3:
                state = torch.FloatTensor(state).unsqueeze(0) # Adiciona dimensão para o batch
            q_values = self.network(state) # Predições de Q-valores para cada ação
            action_idx =  torch.argmax(q_values, dim=1).item() # Indice da ação com maior Q-valor

            # Converte o indice para um vetor de botões (one-hot encoding ou mapeamento especifico)
            action = np.zeros(self.action_dim, dtype=int)
            action[action_idx] = 1 # Define o botão correspondente como pressionado
        
        return action
        
        
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return # Espera encher o buffer
        
        batch = self.sample_experience(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # for i, state in enumerate(states):
        #     print(f'Estado {i} tem shape: {np.array(state).shape}')

        states = np.array([np.array(state, dtype=np.float32) for state in states])
        next_states = np.array([np.array(next_state, dtype=np.float32) for next_state in next_states])

        # Converte os dados para tensores do PyTorch
        states = torch.FloatTensor(np.array(states).squeeze(1))
        actions = np.array(actions)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states).squeeze(1))
        dones = torch.FloatTensor(np.array(dones))

        # Calcula os valores Q para os valores atual
        q_values = self.network(states)
        actions = actions.argmax(dim=1).unsqueeze(1)
        # print(f'q_values sahpe: {q_values.shape}, actions shape: {actions.shape}')
        q_values = q_values.gather(1, actions)

        #calcula os valores Q esperados para o proximo estado
        next_q_values = self.network(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        if len(expected_q_values.shape) > 1 and expected_q_values.shape[1] == 1:
            expected_q_values = expected_q_values.squeeze(1)
        
        # print(f"q_values shape após gather: {q_values.shape}")
        # print(f"expected_q_values shape: {expected_q_values.shape}")

        # Calcula a perda e faz otimização
        loss = F.mse_loss(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Atualiza a taxa de exploração
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
