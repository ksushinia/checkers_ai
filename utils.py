# utils.py

import torch
import numpy as np
import config

def get_nnet_input_tensor(board, player):
    """
    Готовит тензор для подачи в нейросеть из состояния доски.
    """
    channels = np.zeros((4, config.BOARD_Y, config.BOARD_X))
    channels[0][board == player] = 1        # Свои пешки
    channels[1][board == player * 2] = 1    # Свои дамки
    channels[2][board == -player] = 1       # Пешки врага
    channels[3][board == -player * 2] = 1   # Дамки врага
    return torch.FloatTensor(channels).unsqueeze(0)