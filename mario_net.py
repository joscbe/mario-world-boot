import torch
import torch.nn as nn
import torch.nn.functional as F

class MarioNet(nn.Module):
    def __init__(self, action_dim):
        super(MarioNet, self).__init__()

        # camadas concolucionais
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Camada totalmente conectada (linear) para mapear as ativações para as ações
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        # Passa o input pelas camadas convolucionais
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flattening: transforma o output das convoluções em uma única linha de dados
        x = x.view(x.size(0), -1)

        # Passa o output pelas camadas totalemente conectadas
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x