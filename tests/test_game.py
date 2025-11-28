# tests/test_game.py

import sys
import os
import numpy as np

# Эта конструкция позволяет нашему тестовому файлу 'видеть' файлы в родительской директории (где лежит game.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game import CheckersGame


def test_initial_board_and_moves():
    """ Тест: начальная позиция и первые ходы белых. """
    print("--- Запуск теста: Начальная позиция ---")
    game = CheckersGame()

    # Проверяем, что на доске правильное количество шашек
    assert np.count_nonzero(game.board == 1) == 12, "Должно быть 12 белых шашек"
    assert np.count_nonzero(game.board == -1) == 12, "Должно быть 12 черных шашек"

    # Проверяем количество стартовых ходов
    moves = game.get_valid_moves(1)
    assert len(moves) == 7, "В начальной позиции должно быть 7 ходов для белых"
    print(">>> Тест НАЧАЛЬНОЙ ПОЗИЦИИ пройден!\n")


def test_pawn_capture():
    """ Тест: простое взятие пешкой. """
    print("--- Запуск теста: Взятие пешкой ---")
    game = CheckersGame()
    game.board = np.zeros((8, 8))
    game.board[5, 4] = 1  # Белая
    game.board[4, 3] = -1  # Черная

    moves = game.get_valid_moves(1)
    assert len(moves) == 1, "Должен быть найден 1 ход со взятием"
    assert moves[0] == [(5, 4), (3, 2)], "Неверный путь взятия"
    print(">>> Тест ВЗЯТИЯ ПЕШКОЙ пройден!\n")


def test_king_capture_and_game_over():
    """ Тест: взятие дамкой и проверка завершения игры. """
    print("--- Запуск теста: Взятие дамкой и конец игры ---")
    game = CheckersGame()
    game.board = np.zeros((8, 8))
    game.board[1, 1] = 2  # Белая дамка
    game.board[3, 3] = -1  # Единственная черная пешка

    # Перед ходом игра продолжается
    assert game.check_game_over(1) == 0

    move = game.get_valid_moves(1)[0]
    game.make_move(move)

    # После хода у черных нет фигур, они должны проиграть
    # check_game_over(-1) вернет -1, так как для черных нет ни фигур, ни ходов
    assert game.check_game_over(-1) == -1
    print(">>> Тест ВЗЯТИЯ ДАМКОЙ и КОНЦА ИГРЫ пройден!\n")


def test_mini_game_sequence():
    """ Тест: последовательность из нескольких ходов на изолированной доске с выводом доски. """
    print("--- Запуск теста: Мини-игра ---")
    game = CheckersGame()
    # Создаем чистую доску для полного контроля
    game.board = np.zeros((8, 8))

    # Расставляем фигуры для нашего сценария
    game.board[5, 2] = 1  # Белая 1
    game.board[2, 1] = -1  # Черная 1

    print("Начальная расстановка для теста:")
    print(game.board)

    # --- Ход 1: Белые ---
    print("\nХод 1 (Белые): (5, 2) -> (4, 1)")
    game.make_move([(5, 2), (4, 1)])
    print("Доска после хода белых:")
    print(game.board)
    assert game.board[5, 2] == 0 and game.board[4, 1] == 1

    # --- Ход 2: Черные ---
    print("\nХод 2 (Черные): (2, 1) -> (3, 2)")
    game.make_move([(2, 1), (3, 2)])
    print("Доска после хода черных (ситуация для взятия):")
    print(game.board)
    assert game.board[2, 1] == 0 and game.board[3, 2] == -1

    # --- Ход 3: Белые делают взятие ---
    print("\nХод 3 (Белые): Взятие")
    moves = game.get_valid_moves(1)

    assert len(moves) == 1, f"Ожидался 1 ход, но найдено {len(moves)}: {moves}"
    assert moves[0] == [(4, 1), (2, 3)]

    game.make_move(moves[0])
    print("Финальная доска после взятия:")
    print(game.board)
    assert game.board[4, 1] == 0 and game.board[3, 2] == 0 and game.board[2, 3] == 1

    print("\n>>> Тест МИНИ-ИГРЫ пройден!\n")


def test_mandatory_capture_with_bystanders():
    """
    Тест: Проверка правила обязательного взятия, когда на доске есть
    и другие фигуры с простыми ходами.
    """
    print("--- Запуск теста: Обязательное взятие при наличии других ходов ---")
    game = CheckersGame()
    game.board = np.zeros((8, 8))

    # Расставляем фигуры:
    # У белой шашки (4, 3) есть взятие
    game.board[4, 3] = 1  # Белая шашка A
    game.board[3, 4] = -1  # Черная шашка (цель)

    # У белой шашки (5, 0) есть простой ход, который должен быть проигнорирован
    game.board[5, 0] = 1  # Белая шашка B

    # Просто еще одна черная шашка, чтобы доска не была пустой
    game.board[2, 1] = -1

    print("Начальная доска для теста 'Обязательное взятие':")
    print(game.board)

    # Получаем ходы для белых
    moves = game.get_valid_moves(1)

    print(f"\nНайденные ходы для белых: {moves}")

    # Проверяем, что найден ТОЛЬКО один ход - взятие
    assert len(moves) == 1, f"Ожидался 1 обязательный ход, но найдено {len(moves)}. Простой ход не был проигнорирован."

    # Проверяем, что этот ход - именно тот, который мы ожидали
    expected_move = [(4, 3), (2, 5)]
    assert moves[0] == expected_move, f"Ожидался ход {expected_move}, но получен {moves[0]}"

    print("Проверка успешна: движок правильно проигнорировал простой ход и оставил только взятие.")

    # Выполняем ход для завершения теста
    game.make_move(moves[0])
    print("\nДоска после выполнения обязательного взятия:")
    print(game.board)
    assert game.board[3, 4] == 0 and game.board[2, 5] == 1

    print("\n>>> Тест ОБЯЗАТЕЛЬНОГО ВЗЯТИЯ пройден!\n")


def test_mid_game_scenario_with_many_pieces():
    """
    Тест: Последовательность ходов в середине игры (12 шашек)
    с финальной проверкой обязательного взятия.
    """
    print("--- Запуск теста: Сценарий середины игры (12 шашек) ---")
    game = CheckersGame()
    game.board = np.zeros((8, 8))

    # Расставляем 6 белых и 6 черных шашек для сценария
    game.board[7, 0] = 1;
    game.board[6, 1] = 1;
    game.board[5, 2] = 1
    game.board[7, 4] = 1;
    game.board[6, 5] = 1;
    game.board[5, 6] = 1
    game.board[0, 1] = -1;
    game.board[1, 2] = -1;
    game.board[2, 3] = -1
    game.board[0, 5] = -1;
    game.board[1, 6] = -1;
    game.board[2, 7] = -1

    print("Начальная доска для 'мид-гейм' сценария:")
    print(game.board)

    # --- Ход 1: Белые ---
    print("\nХод 1 (Белые): (5, 2) -> (4, 3)")
    game.make_move([(5, 2), (4, 3)])

    # --- Ход 2: Черные ---
    print("\nХод 2 (Черные): (2, 3) -> (3, 4)")
    game.make_move([(2, 3), (3, 4)])
    print("Доска после хода черных (создана ситуация для ДВОЙНОГО взятия):")
    print(game.board)

    # --- Ход 3: Проверка для Белых ---
    print("\nХод 3 (Белые): Проверка обязательного взятия")
    moves = game.get_valid_moves(1)

    print(f"Найденные ходы для белых: {moves}")
    assert len(moves) == 1, "Движок должен был найти только один обязательный ход!"

    # ИСПРАВЛЕНО: Теперь мы ожидаем правильную, длинную цепочку взятия
    expected_move = [(4, 3), (2, 5), (0, 7)]
    assert moves[0] == expected_move, f"Найден неверный путь для взятия. Ожидался {expected_move}"

    print("Проверка успешна: найден единственно верный ход (двойное взятие).")

    # Выполняем этот ход
    game.make_move(moves[0])
    print("\nФинальная доска после взятия:")
    print(game.board)

    # ИСПРАВЛЕНО: Проверяем, что обе черные шашки сбиты, а белая стала ДАМКОЙ (2) на последнем ряду
    assert game.board[3, 4] == 0, "Первая сбитая шашка не удалена"
    assert game.board[1, 6] == 0, "Вторая сбитая шашка не удалена"
    assert game.board[0, 7] == 2, "Шашка должна была стать дамкой (2) на финальной позиции"

    print("\n>>> Тест СЦЕНАРИЯ СЕРЕДИНЫ ИГРЫ пройден!\n")


if __name__ == '__main__':
    test_initial_board_and_moves()
    test_pawn_capture()
    test_king_capture_and_game_over()
    test_mini_game_sequence()
    test_mandatory_capture_with_bystanders()
    # Добавляем вызов нового теста
    test_mid_game_scenario_with_many_pieces()
    print("====== ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ! ======")