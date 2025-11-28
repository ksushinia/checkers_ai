# inspector.py
import torch

CHECKPOINT_FILE = '1-1000-checkpoint.tar'  # Укажи путь к твоему файлу

print(f"--- Анализ файла: {CHECKPOINT_FILE} ---")

try:
    # Загружаем файл на CPU, чтобы не требовать GPU
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=torch.device('cpu'), weights_only=False)

    print("\n[Ключи, найденные в чекпоинте]:")
    for key in checkpoint.keys():
        print(f"- {key}")

    if 'iteration' in checkpoint:
        print(f"\n[Итерация]: Модель обучена до конца {checkpoint['iteration']}-й итерации.")

    if 'state_dict' in checkpoint:
        print("\n[Архитектура модели (state_dict)]:")
        # Выведем первые 5 слоев для примера
        for i, (name, params) in enumerate(checkpoint['state_dict'].items()):
            print(f" - Слой '{name}' имеет размер: {params.shape}")
            if i >= 4:
                break

    if 'history' in checkpoint:
        print(f"\n[История обучения]: Сохранено {len(checkpoint['history'])} игровых примеров.")

except FileNotFoundError:
    print("Файл чекпоинта не найден.")
except Exception as e:
    print(f"Произошла ошибка при чтении файла: {e}")