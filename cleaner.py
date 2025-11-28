import torch
import os

# --- НАСТРОЙКИ ---
INPUT_FILE = 'tars/Exp2/Exp2-400.tar'  # Твой тяжелый файл
OUTPUT_FILE = 'tars/Exp2/little/Exp2-400-lite.tar'  # Имя для легкого файла


def clean_checkpoint():
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден!")
        return

    print(f"1. Загружаю тяжелый файл (это займет время)... {INPUT_FILE}")
    # map_location='cpu' важен, чтобы не забивать видеопамять, если она есть
    checkpoint = torch.load(INPUT_FILE, map_location='cpu', weights_only=False)

    print("2. Извлекаю только веса нейросети...")

    # Обычно веса лежат под ключом 'state_dict', судя по твоему коду
    if 'state_dict' in checkpoint:
        model_weights = checkpoint['state_dict']

        # Создаем новый словарь только с весами
        new_checkpoint = {'state_dict': model_weights}

        print(f"3. Сохраняю легкую версию в {OUTPUT_FILE}...")
        torch.save(new_checkpoint, OUTPUT_FILE)

        # Проверка размера
        old_size = os.path.getsize(INPUT_FILE) / (1024 * 1024)
        new_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"Готово! Размер уменьшен с {old_size:.2f} MB до {new_size:.2f} MB")
    else:
        print("ОШИБКА: В файле не найден ключ 'state_dict'. Структура файла отличается.")
        print("Доступные ключи:", checkpoint.keys())


if __name__ == "__main__":
    clean_checkpoint()