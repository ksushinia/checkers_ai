# plotter.py

import re
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_FILE = 'training.log'  # Имя файла, откуда читаем данные


def parse_log():
    """Читает ВЕСЬ лог-файл и извлекает данные."""

    if not os.path.exists(LOG_FILE):
        print(f"Лог-файл {LOG_FILE} не найден.")
        return [], [], [], []

    game_len_regex = re.compile(r".*Завершена. Собрано (\d+) примеров.")
    loss_regex = re.compile(r".*Потери \(Loss\): Value=([\d.]+), Policy=([\d.]+)")

    iterations, avg_game_lengths, policy_losses, value_losses = [], [], [], []
    current_iter_lengths = []

    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            game_len_match = game_len_regex.match(line)
            if game_len_match:
                current_iter_lengths.append(int(game_len_match.group(1)))
                continue

            loss_match = loss_regex.search(line)
            if loss_match and current_iter_lengths:
                avg_game_lengths.append(np.mean(current_iter_lengths))
                value_losses.append(float(loss_match.group(1)))
                policy_losses.append(float(loss_match.group(2)))
                iterations.append(len(iterations) + 1)
                current_iter_lengths = []

    return iterations, avg_game_lengths, policy_losses, value_losses


def create_plots(iterations, avg_game_lengths, policy_losses, value_losses, folder, title_prefix=''):
    """Создает и сохраняет набор из 3 графиков в указанную папку."""

    if not iterations:
        print("Нет данных для построения графиков.")
        return

    os.makedirs(folder, exist_ok=True)

    # График 1: Policy Loss
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, policy_losses, marker='o', linestyle='-')
    plt.title(f'{title_prefix} Policy Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'policy_loss.png'))
    plt.close()

    # График 2: Value Loss
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, value_losses, marker='o', linestyle='-', color='r')
    plt.title(f'{title_prefix} Value Loss over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'value_loss.png'))
    plt.close()

    # График 3: Средняя длина игр
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, avg_game_lengths, marker='o', linestyle='-', color='g')
    plt.title(f'{title_prefix} Average Game Length over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Moves')
    plt.grid(True)
    plt.savefig(os.path.join(folder, 'avg_game_length.png'))
    plt.close()

    print(f"Графики успешно сохранены в папку '{folder}'.")


def plot_for_milestone(experiment_name, milestone_iter):
    """Главная функция. Строит общий и поэтапный графики."""

    # 1. Читаем все данные из лога
    all_its, all_lengths, all_p_losses, all_v_losses = parse_log()

    # --- НОВОЕ УМНОЕ ОПРЕДЕЛЕНИЕ ПУТИ ---
    # Проверяем, запущена ли программа в среде Google Colab
    if os.path.exists('/content/drive'):
        base_plot_folder = '/content/drive/MyDrive/CheckersAI/plots'
    else:
        base_plot_folder = 'plots'

    # 2. Создаем папку для эксперимента, используя правильный базовый путь
    exp_folder = os.path.join(base_plot_folder, experiment_name)
    # --- КОНЕЦ НОВОГО БЛОКА ---

    # 3. Строим ОБЩИЙ график по всем данным
    print("Построение ОБЩЕГО графика...")
    create_plots(all_its, all_lengths, all_p_losses, all_v_losses, exp_folder,
                 title_prefix=f"{experiment_name} - Overall")

    # 4. Строим ПОЭТАПНЫЙ график для последнего блока в 100 итераций
    print("Построение ПОЭТАПНОГО графика...")
    start_index = max(0, milestone_iter - 100)
    end_index = milestone_iter

    # "Вырезаем" нужный кусок данных
    stage_its = all_its[start_index:end_index]
    stage_lengths = all_lengths[start_index:end_index]
    stage_p_losses = all_p_losses[start_index:end_index]
    stage_v_losses = all_v_losses[start_index:end_index]

    # Создаем подпапку для этого этапа
    stage_folder = os.path.join(exp_folder, f"iter_{start_index + 1}_to_{end_index}")
    create_plots(stage_its, stage_lengths, stage_p_losses, stage_v_losses, stage_folder,
                 title_prefix=f"{experiment_name} - Stage")


if __name__ == '__main__':
    # Пример вызова из командной строки: python plotter.py Exp2_FineTune 200
    import sys

    if len(sys.argv) == 3:
        exp_name = sys.argv[1]
        milestone = int(sys.argv[2])
        plot_for_milestone(exp_name, milestone)
    else:
        print("Использование: python plotter.py <ИмяЭксперимента> <НомерИтерации>")