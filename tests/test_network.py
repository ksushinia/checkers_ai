# tests/test_network.py

import sys
import os
import torch
import numpy as np

# Добавляем корневую папку проекта в путь, чтобы найти game, network, config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game import CheckersGame
from network import CheckersNet
import config


def get_nnet_input_tensor(board, player):
    """
    Готовит тензор для подачи в нейросеть из состояния доски.
    """
    channels = np.zeros((4, config.BOARD_Y, config.BOARD_X))
    channels[0][board == player] = 1  # Свои пешки
    channels[1][board == player * 2] = 1  # Свои дамки
    channels[2][board == -player] = 1  # Пешки врага
    channels[3][board == -player * 2] = 1  # Дамки врага
    return torch.FloatTensor(channels).unsqueeze(0)


def test_network_io():
    """
    Тест: Проверяет, что нейросеть принимает на вход тензор доски
    и возвращает тензоры политики и ценности правильной размерности.
    """
    print("\n--- Запуск теста для network.py ---")

    # 1. Инициализация
    game = CheckersGame()
    nnet = CheckersNet()

    # 2. Получаем состояние доски
    board_state = game.get_board_state()

    # 3. Превращаем доску в тензор
    input_tensor = get_nnet_input_tensor(board_state, player=1)

    # 4. Проверяем размерность входа
    print(f"Размер входного тензора: {input_tensor.shape}")
    assert input_tensor.shape == (1, 4, 8, 8)

    # 5. Подаем тензор в нейросеть
    nnet.eval()  # Переключаем модель в режим оценки
    policy_logits, value = nnet(input_tensor)

    # 6. Проверяем размерность выходов
    print(f"Размер выхода 'политики': {policy_logits.shape}")
    print(f"Размер выхода 'ценности': {value.shape}")

    assert policy_logits.shape == (1, config.ACTION_SIZE)
    assert value.shape == (1, 1)

    print(">>> Тест НЕЙРОСЕТИ пройден! Критерий выполнения этапа 2 достигнут.")


def test_input_tensor_representation():
    """
    Тест: Проверяет, что get_nnet_input_tensor правильно раскладывает
    все типы фигур по 4-м каналам.
    """
    print("\n--- Запуск теста: Корректность представления доски в тензоре ---")
    board = np.zeros((8, 8))

    # Расставляем все 4 типа фигур с точки зрения игрока 1 (Белые)
    board[1, 0] = 1  # Своя пешка
    board[1, 2] = 2  # Своя дамка
    board[6, 1] = -1  # Вражеская пешка
    board[6, 3] = -2  # Вражеская дамка

    player = 1

    # Получаем 4-канальный тензор
    input_tensor = get_nnet_input_tensor(board, player)

    # Тензор имеет форму (1, 4, 8, 8). Уберем первое измерение для удобства.
    channels = input_tensor.squeeze(0)

    # Проверяем каждый канал: в нем должна быть только одна единица в нужном месте

    # Канал 0: Свои пешки
    assert torch.sum(channels[0]) == 1 and channels[0, 1, 0] == 1
    print("Канал 0 (Свои пешки) - ОК")

    # Канал 1: Свои дамки
    assert torch.sum(channels[1]) == 1 and channels[1, 1, 2] == 1
    print("Канал 1 (Свои дамки) - ОК")

    # Канал 2: Пешки врага
    assert torch.sum(channels[2]) == 1 and channels[2, 6, 1] == 1
    print("Канал 2 (Пешки врага) - ОК")

    # Канал 3: Дамки врага
    assert torch.sum(channels[3]) == 1 and channels[3, 6, 3] == 1
    print("Канал 3 (Дамки врага) - ОК")

    print(">>> Тест ПРЕДСТАВЛЕНИЯ ДАННЫХ пройден!")


if __name__ == '__main__':
    test_network_io()
    # Добавляем вызов нового теста
    test_input_tensor_representation()