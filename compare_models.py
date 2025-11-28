# compare_models.py
import torch
import os

# имена файлов для сравнения
CHECKPOINT_A_NAME = "tars/Exp1/Exp1-100.tar"
CHECKPOINT_B_NAME = "tars/Exp1/Exp1-500.tar"  #

# Если файлы лежат в той же папке '.'
# Если в Colab: '/content/drive/MyDrive/CheckersAI'
CHECKPOINT_FOLDER = '.'


def analyze_checkpoint(filename):
    """Загружает один чекпоинт и выводит его параметры."""

    filepath = os.path.join(CHECKPOINT_FOLDER, filename)
    print("\n" + "=" * 50)
    print(f"Анализ файла: {filename}")
    print("=" * 50)

    try:
        # Загружаем на CPU, чтобы не требовать GPU
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)

        # 1. Номер итерации
        if 'iteration' in checkpoint:
            iteration_num = checkpoint['iteration'] + 1
            print(f"  [Итерация]: Модель обучена до {iteration_num}-й итерации.")
        else:
            print("  [Итерация]: Номер итерации не найден (старый формат).")

        # 2. Размер "памяти" (истории игр)
        if 'history' in checkpoint:
            history_size = len(checkpoint['history'])
            print(f"  [История]: Хранит {history_size} игровых примеров.")

        # 3. Ключевые параметры оптимизатора (скорость обучения)
        if 'optimizer' in checkpoint:
            lr = checkpoint['optimizer']['param_groups'][0]['lr']
            print(f"  [Скорость обучения (LR)]: {lr}")

        # 4. Анализ весов (просто для интереса)
        # Посчитаем среднее значение весов первого слоя - это может косвенно
        # указывать на то, "насытилась" ли сеть.
        if 'state_dict' in checkpoint:
            first_layer_weights = checkpoint['state_dict']['conv1.weight']
            print(f"  [Веса]: Среднее значение весов 1-го слоя: {first_layer_weights.mean():.6f}")

    except FileNotFoundError:
        print("!!! Ошибка: Файл не найден.")
    except Exception as e:
        print(f"!!! Произошла ошибка при чтении файла: {e}")


if __name__ == '__main__':
    analyze_checkpoint(CHECKPOINT_A_NAME)
    analyze_checkpoint(CHECKPOINT_B_NAME)