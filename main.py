# main.py

import sys
import time
import torch
import numpy as np
import subprocess

import config
from game import CheckersGame
from network import CheckersNet
from mcts import MCTS
from train import Trainer
from plotter import parse_log, create_plots


# --- Управление логгированием ---
class Logger:
    def __init__(self, filename="training.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')  # 'w' - перезаписывать файл при каждом запуске

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# --- Главная логика ---

def execute_episode(game, nnet):
    """Проигрывает одну полную игру и возвращает обучающие данные."""
    game_history = []
    episode_game = CheckersGame()
    current_player = 1
    turn = 0
    MAX_TURNS = 150
    mcts = MCTS(episode_game, nnet)

    while True:
        turn += 1
        if turn > MAX_TURNS:
            return [(x[0], x[2], 1e-4) for x in game_history]

        canonical_board = episode_game.get_canonical_form(episode_game.board, current_player)
        pi = mcts.getActionProb(canonical_board, turn=turn, temp=(1 if turn < 30 else 0))
        game_history.append([canonical_board, current_player, pi])

        action = np.random.choice(len(pi), p=pi)

        next_board, next_player = episode_game.get_next_state(episode_game.board, current_player, action)
        episode_game.board = next_board
        current_player = next_player

        game_result = episode_game.check_game_over(current_player)

        if game_result != 0:
            return [(x[0], x[2], game_result * ((-1) ** (x[1] != current_player))) for x in game_history]


def main():
    """Главный дирижер."""

    sys.stdout = Logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    game = CheckersGame()
    nnet = CheckersNet().to(device)
    trainer = Trainer(nnet, device)

    start_iter = 0
    # Если в конфиге указан файл для старта, загружаем его и обнуляем счетчик.
    if config.LOAD_CHECKPOINT_FILE:
        print(f"Загрузка с указанного чекпоинта: {config.LOAD_CHECKPOINT_FILE}")
        # Передаем имя файла напрямую
        start_iter = trainer.load_checkpoint(filename=config.LOAD_CHECKPOINT_FILE)
        start_iter = 0  # Начинаем новый эксперимент с 0
    else:
        # Иначе, пытаемся возобновить текущий эксперимент
        print(f"Попытка возобновить эксперимент '{config.EXPERIMENT_NAME}'...")
        # Передаем только имя эксперимента
        start_iter = trainer.load_checkpoint(experiment_name=config.EXPERIMENT_NAME)

    LOG_INTERVAL = 25
    start_time = time.time()

    for i in range(start_iter, config.NUM_ITERATIONS):
        print(f"Итерация {i + 1}/{config.NUM_ITERATIONS}")

        # ... (код Self-Play) ...
        training_examples = []
        for g in range(config.NUM_GAMES):
            print(f"  Игра {g + 1}/{config.NUM_GAMES}...")
            new_examples = execute_episode(game, trainer.nnet)
            training_examples.extend(new_examples)
            print(f"  ...Завершена. Собрано {len(new_examples)} примеров.")

        # ... (код Обучения) ...
        print("\n--- Этап обучения ---")
        trainer.train(training_examples)

        # ... (код Сохранения) ...
        print("Сохранение чекпоинта...")
        trainer.save_checkpoint(i, config.EXPERIMENT_NAME)  # Передаем имя эксперимента
        print("-" * 30)

        # --- ИЗМЕНЕНИЕ №2: Логика построения графиков ---
        if (i + 1) % 100 == 0 and i >= start_iter:
            elapsed_time = time.time() - start_time
            print(f"\n[ЛОГ ВРЕМЕНИ] 100 итераций завершены за {elapsed_time:.2f} секунд.")

            print("Построение графиков прогресса...")
            try:
                # Передаем имя эксперимента и номер итерации
                subprocess.run(['python', 'plotter.py', config.EXPERIMENT_NAME, str(i + 1)])
            except FileNotFoundError:
                print("Не удалось запустить plotter.py.")
            start_time = time.time()

    print("\n\nОБУЧЕНИЕ ПОЛНОСТЬЮ ЗАВЕРШЕНО!\n")
    print("Построение финальных итоговых графиков...")
    try:
        # Передаем имя и ОБЩЕЕ количество итераций
        subprocess.run(['python', 'plotter.py', config.EXPERIMENT_NAME, str(config.NUM_ITERATIONS)])
    except FileNotFoundError:
        print("Не удалось запустить plotter.py.")

    print("\nФинальные графики сохранены. Проект успешно завершен.")


if __name__ == "__main__":
    main()