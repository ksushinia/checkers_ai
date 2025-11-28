# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Та самая "остаточная связь"
        out = F.relu(out)
        return out


class CheckersNet(nn.Module):
    def __init__(self):
        super(CheckersNet, self).__init__()
        # Параметры сети
        self.input_channels = 4
        self.num_channels = 128  # Количество фильтров
        self.num_res_blocks = 5  # Количество блоков (можно увеличить до 10-20 при наличии GPU)

        # Начальный сверточный слой
        self.conv_input = nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(self.num_channels)

        # Башня из ResNet блоков
        self.res_tower = nn.Sequential(
            *[ResidualBlock(self.num_channels) for _ in range(self.num_res_blocks)]
        )

        # "Голова" Политики
        self.policy_conv = nn.Conv2d(self.num_channels, 32, kernel_size=1)  # Уменьшаем каналы перед полносвязным
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * config.BOARD_X * config.BOARD_Y, config.ACTION_SIZE)

        # "Голова" Ценности
        self.value_conv = nn.Conv2d(self.num_channels, 3, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * config.BOARD_X * config.BOARD_Y, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Входная свертка
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Проход через ResNet башню
        x = self.res_tower(x)

        # Голова политики
        pi = F.relu(self.policy_bn(self.policy_conv(x)))
        pi = pi.view(-1, 32 * config.BOARD_X * config.BOARD_Y)
        pi = self.policy_fc(pi)
        pi = F.log_softmax(pi, dim=1)

        # Голова ценности
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 3 * config.BOARD_X * config.BOARD_Y)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return pi, v

    def predict(self, board_tensor):
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board_tensor)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]