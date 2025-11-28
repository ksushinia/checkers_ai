# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
import _pickle

import config
from utils import get_nnet_input_tensor

# --- НАЧАЛО "УМНОГО" БЛОКА ---
# Проверяем, запущена ли программа в среде Google Colab
IN_COLAB = os.path.exists('/content/drive')

if IN_COLAB:
    # Если да, определяем путь к папке в Google Drive
    print("Обнаружена среда Google Colab. Чекпоинты будут сохраняться в Google Drive.")
    CHECKPOINT_FOLDER = '/content/drive/MyDrive/CheckersAI'
else:
    # Если нет, сохраняем в текущую папку проекта
    print("Среда Google Colab не обнаружена. Чекпоинты будут сохраняться локально.")
    CHECKPOINT_FOLDER = '.' # '.' означает "текущая папка"


class Trainer:
    def __init__(self, nnet, device):
        self.nnet = nnet
        self.device = device
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=config.LEARNING_RATE)
        # Было: maxlen=5 * config.NUM_GAMES
        # Стало: Храним историю за последние ~20-50 итераций (или просто фиксированное число)
        self.training_data_history = deque(maxlen=40000)

    def train(self, examples):
        """
        Обучает нейросеть на предоставленных примерах.
        `examples` - это список из троек: (доска, вектор_pi, результат_v)
        """

        # --- ФИНАЛЬНЫЙ, БРОНЕБОЙНЫЙ ФИКС ---
        # Перед началом обучения мы в любом случае принудительно
        # отправляем модель на нужное устройство. Это защитит нас
        # от любых "призрачных" перемещений.
        self.nnet.to(self.device)
        # ------------------------------------

        self.training_data_history.extend(examples)
        self.nnet.train()

        for epoch in range(config.EPOCHS):
            print(f"    Эпоха {epoch + 1}/{config.EPOCHS}")

            # Если данных меньше, чем размер батча, используем все, что есть
            batch_size = min(len(self.training_data_history), config.BATCH_SIZE)
            if batch_size == 0:
                continue  # Пропускаем эпоху, если данных нет

            training_batch = random.sample(self.training_data_history, batch_size)

            # Разделяем данные на компоненты
            boards = [data[0] for data in training_batch]
            target_pis = [data[1] for data in training_batch]
            target_vs = [data[2] for data in training_batch]

            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Создаем тензоры и СРАЗУ отправляем их на нужное устройство (GPU/CPU)
            board_tensors = torch.cat([get_nnet_input_tensor(b, 1) for b in boards], dim=0).to(self.device)
            target_pis = torch.FloatTensor(np.array(target_pis)).to(self.device)
            target_vs = torch.FloatTensor(np.array(target_vs).astype(np.float64)).to(self.device)

            # --- Прямой проход (Forward pass) ---
            out_pis, out_vs = self.nnet(board_tensors)

            # --- Расчет функций потерь (Loss calculation) ---
            loss_v = F.mse_loss(out_vs.view(-1), target_vs)
            loss_pi = -torch.sum(target_pis * out_pis) / target_pis.size()[0]
            total_loss = loss_pi + loss_v

            # --- Обратное распространение и шаг оптимизатора ---
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Выводим потери после последней эпохи
        print(f"  > Потери (Loss): Value={loss_v.item():.4f}, Policy={loss_pi.item():.4f}")

    def save_checkpoint(self, i, experiment_name):
        """ Сохраняет текущий и промежуточный чекпоинты. """
        os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

        # 1. Сохраняем "аварийный" чекпоинт для возобновления
        current_filepath = os.path.join(CHECKPOINT_FOLDER, f"{experiment_name}-current.tar")
        torch.save({
            'iteration': i,
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.training_data_history,
        }, current_filepath)

        # 2. Каждые 100 итераций сохраняем "исторический" чекпоинт
        if (i + 1) % 100 == 0:
            milestone_filepath = os.path.join(CHECKPOINT_FOLDER, f"{experiment_name}-{i + 1}.tar")
            torch.save({
                'iteration': i,
                'state_dict': self.nnet.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'history': self.training_data_history,
            }, milestone_filepath)
            print(f"Сохранен исторический чекпоинт: {milestone_filepath}")

    def load_checkpoint(self, experiment_name=None, filename=None):
        """ Загружает либо указанный файл, либо "аварийный" чекпоинт эксперимента. """
        if filename:
            filepath = os.path.join(CHECKPOINT_FOLDER, filename)
        elif experiment_name:
            filepath = os.path.join(CHECKPOINT_FOLDER, f"{experiment_name}-current.tar")
        else:
            # Если не указано ни то, ни другое, ничего не делаем
            print("Не указан файл или имя эксперимента для загрузки. Начинаем с нуля.")
            return 0

        try:
            # --- ГЛАВНОЕ ИЗМЕНЕНИЕ ЗДЕСЬ ---
            # Явно указываем weights_only=False, чтобы разрешить загрузку объекта 'deque'
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

            self.nnet.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.nnet.to(self.device)

            if 'history' in checkpoint:
                self.training_data_history = checkpoint['history']

            print(f"Загружен чекпоинт: {filepath}")

            if 'iteration' in checkpoint:
                return checkpoint['iteration'] + 1
            else:
                print("Внимание: в чекпоинте отсутствует номер итерации. Начинаем с 0.")
                return 0

        except FileNotFoundError:
            print(f"Чекпоинт не найден: {filepath}. Начинаем с нуля.")
            return 0

        # --- ДОБАВЛЕНА ОБРАБОТКА НОВОЙ ОШИБКИ ---
        except _pickle.UnpicklingError:
            print(f"Ошибка загрузки чекпоинта: {filepath}. Возможно, он поврежден или несовместим. Начинаем с нуля.")
            return 0