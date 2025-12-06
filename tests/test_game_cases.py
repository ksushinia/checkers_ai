import numpy as np
import sys
import copy

# Импортируем реальные модули проекта
import config
from game import CheckersGame


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'


def get_board_lines(board):
    symbols = {0: '.', 1: 'w', -1: 'b', 2: 'W', -2: 'B'}
    lines = []
    lines.append("   0 1 2 3 4 5 6 7 ")
    lines.append("  ------------------")
    for r in range(config.BOARD_Y):
        row_str = f"{r} |"
        for c in range(config.BOARD_X):
            val = board[r, c]
            sym = symbols.get(val, '?')
            # Раскраска фигур
            if val > 0:  # Белые
                sym = f"{Colors.OKGREEN}{sym}{Colors.ENDC}"
            elif val < 0:  # Черные
                sym = f"{Colors.FAIL}{sym}{Colors.ENDC}"
            row_str += f"{sym} "
        row_str += "|"
        lines.append(row_str)
    lines.append("  ------------------")
    return lines


def print_comparison(start_board, target_board, title_start="ДО ХОДА", title_end="ОЖИДАНИЕ"):
    lines_a = get_board_lines(start_board)
    lines_b = get_board_lines(target_board)

    print(f"\n{title_start.center(20)}       {title_end.center(20)}")
    for la, lb in zip(lines_a, lines_b):
        print(f"{la}     {lb}")
    print()


def create_board_from_pieces(pieces):
    board = np.zeros((config.BOARD_X, config.BOARD_Y))
    for r, c, val in pieces:
        board[r, c] = val
    return board


def run_test_case(test_name, start_pieces, target_pieces, player, expect_valid=True):
    print(f"{Colors.HEADER} ТЕСТ: {test_name}{Colors.ENDC}")

    game = CheckersGame()

    # инициализация
    start_board = create_board_from_pieces(start_pieces)
    target_board = create_board_from_pieces(target_pieces)
    game.board = np.copy(start_board)

    # визуализация
    print_comparison(start_board, target_board,
                     title_start=f" Игрок {player} (Start)",
                     title_end="Цель (Target)")

    # запрашиваем ходы
    valid_moves = game.get_valid_moves(player)

    matched_move = None

    print(f"{Colors.CYAN}Алгоритм нашел {len(valid_moves)} вариантов хода:{Colors.ENDC}")

    for i, move in enumerate(valid_moves):
        # применяем ход
        temp_game = CheckersGame()
        temp_game.board = np.copy(start_board)
        temp_game.make_move(move)

        # формируем строку описания
        path_str = " -> ".join([str(pos) for pos in move])
        if len(move) > 2:
            path_str += f" ({Colors.BOLD}Взятий: {len(move) - 1}{Colors.ENDC})"

        is_match = np.array_equal(temp_game.board, target_board)

        match_mark = f" {Colors.OKGREEN}[MATCH]{Colors.ENDC}" if is_match else ""
        print(f"  {i + 1}. {path_str}{match_mark}")

        if is_match:
            matched_move = move

    # анализ результатов
    success = False
    if expect_valid:
        if matched_move:
            print(f"\n{Colors.OKGREEN} УСПЕХ: Ожидаемый ход найден.{Colors.ENDC}")
            success = True
        else:
            print(f"\n{Colors.FAIL} ОШИБКА: Требуемый ход не найден.{Colors.ENDC}")
    else:
        if matched_move:
            print(f"\n{Colors.FAIL} ОШИБКА: Алгоритм разрешил запрещенный ход!{Colors.ENDC}")
        else:
            print(f"\n{Colors.OKGREEN} УСПЕХ: Запрещенный ход корректно отклонен.{Colors.ENDC}")
            success = True

    #print("-" * 60 + "\n")
    return success


if __name__ == "__main__":
    print(f"{Colors.BOLD}ЗАПУСК ТЕСТОВ{Colors.ENDC}\n")
    results = []

    # ГРУППА 1: Пешки
    results.append(run_test_case(
        "1. Тихий ход белой пешки",
        start_pieces=[(5, 2, 1)],
        target_pieces=[(4, 3, 1)], player=1
    ))

    results.append(run_test_case(
        "2. Запрет выхода за границы",
        start_pieces=[(4, 0, 1)],
        target_pieces=[(4, 7, 1)], player=1, expect_valid=False
    ))

    # ГРУППА 2: Взятия пешками
    results.append(run_test_case(
        "3. Одиночное взятие",
        start_pieces=[(4, 4, 1), (3, 3, -1)],
        target_pieces=[(2, 2, 1)], player=1
    ))

    results.append(run_test_case(
        "4. Взятие назад пешкой",
        start_pieces=[(3, 3, 1), (4, 4, -1)],
        target_pieces=[(5, 5, 1)], player=1
    ))

    results.append(run_test_case(
        "5. Приоритет взятия",
        start_pieces=[(4, 4, 1), (3, 3, -1)],
        target_pieces=[(3, 5, 1), (3, 3, -1)], player=1, expect_valid=False
    ))

    results.append(run_test_case(
        "6. Двойное взятие пешкой",
        start_pieces=[(6, 0, 1), (5, 1, -1), (3, 3, -1)],
        target_pieces=[(2, 4, 1)], player=1
    ))

    results.append(run_test_case(
        "7. Тройное взятие пешкой",
        start_pieces=[(7, 0, 1), (6, 1, -1), (4, 3, -1), (2, 5, -1)],
        target_pieces=[(1, 6, 1)], player=1
    ))

    results.append(run_test_case(
        "8. Запрет прерывания серии",
        start_pieces=[(6, 0, 1), (5, 1, -1), (3, 3, -1)],
        target_pieces=[(4, 2, 1), (3, 3, -1)], player=1, expect_valid=False
    ))

    # ГРУППА 3: Дамки

    results.append(run_test_case(
        "9. Превращение в дамку",
        start_pieces=[(1, 1, 1)],
        target_pieces=[(0, 0, 2)], player=1
    ))

    results.append(run_test_case(
        "10. Тихий ход дамки через все поле",
        start_pieces=[(7, 0, 2)],
        target_pieces=[(0, 7, 2)], player=1
    ))

    results.append(run_test_case(
        "11. Взятие дамкой с остановкой через несколько клеток",
        start_pieces=[(7, 0, 2), (4, 3, -1)],
        target_pieces=[(1, 6, 2)], player=1
    ))

    # ГРУППА 4: Зигзаги

    results.append(run_test_case(
        "12. Множественное взятие дамкой с поворотом",
        start_pieces=[(7, 0, 2), (5, 2, -1), (3, 2, -1)],
        target_pieces=[(2, 1, 2)],
        player=1
    ))

    results.append(run_test_case(
        "13. 5 фигур подряд",
        start_pieces=[
            (7, 0, 2),  # Белая Дамка
            (6, 1, -1),  # Жертва 1
            (4, 1, -1),  # Жертва 2
            (2, 1, -1),  # Жертва 3
            (2, 3, -1),  # Жертва 4
            (4, 5, -1)  # Жертва 5
        ],
        target_pieces=[(5, 6, 2)],  # Финальная позиция дамки
        player=1
    ))

    # ГРУППА 5: Негативные сценарии
    results.append(run_test_case(
        "14. Запрет взятия своих",
        start_pieces=[(4, 4, 1), (3, 3, 1)],
        target_pieces=[(2, 2, 1), (3, 3, 0)], player=1, expect_valid=False
    ))

    results.append(run_test_case(
        "15. Запрет прыжка через воздух",
        start_pieces=[(4, 4, 1)],
        target_pieces=[(2, 2, 1)], player=1, expect_valid=False
    ))

    print(f"{Colors.HEADER} ТЕСТ: 16. Проверка отсутствия ходов{Colors.ENDC}")
    game_pat = CheckersGame()
    game_pat.board = create_board_from_pieces([(1, 0, 1), (0, 1, 1)])
    print_comparison(game_pat.board, game_pat.board, "Ситуация: нет ходов", "Ожидание: 0 ходов")
    moves_pat = game_pat.get_valid_moves(1)

    if len(moves_pat) == 0:
        print(f"{Colors.OKGREEN} УСПЕХ: Ходов нет, список пуст.{Colors.ENDC}")
        results.append(True)
    else:
        print(f"{Colors.FAIL} ОШИБКА: Найдены ходы: {moves_pat}{Colors.ENDC}")
        results.append(False)
    #print("-" * 60 + "\n")

    # ИТОГИ
    print(f"{Colors.BOLD}ИТОГИ:{Colors.ENDC}")
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"{Colors.OKGREEN}ВСЕ ТЕСТЫ ПРОЙДЕНЫ ({passed}/{total}){Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}ЕСТЬ ОШИБКИ! Успешно: {passed}/{total}{Colors.ENDC}")