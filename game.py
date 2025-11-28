# Вся логика игры в шашки
# game.py


import numpy as np
import config

class CheckersGame:
    def __init__(self):
        self.board = np.zeros((config.BOARD_X, config.BOARD_Y))
        # Вызываем метод для расстановки шашек
        self.reset_board()

    def reset_board(self):
        """ Расставляет шашки в начальную позицию. """
        # Сначала очищаем доску
        self.board = np.zeros((config.BOARD_X, config.BOARD_Y))

        # Расставляем черные шашки (-1)
        for r in range(3):
            for c in range(config.BOARD_X):
                if (r + c) % 2 != 0:
                    self.board[r, c] = -1

        # Расставляем белые шашки (1)
        for r in range(config.BOARD_Y - 3, config.BOARD_Y):
            for c in range(config.BOARD_X):
                if (r + c) % 2 != 0:
                    self.board[r, c] = 1

    def get_board_state(self):
        """Возвращает текущее состояние доски."""
        return self.board

    def is_valid_pos(self, r, c):
        """ Проверяет, что координаты (r, c) находятся в пределах доски. """
        return 0 <= r < config.BOARD_Y and 0 <= c < config.BOARD_X

    def get_valid_moves(self, player):
        """
        Возвращает список всех возможных ходов для текущего игрока.
        Формат хода: [(y_start, x_start), (y_end, x_end), (y_captured, x_captured), ...]
        """

        # 1. Сначала ищем все возможные ходы со взятием
        capture_moves = self._find_all_capture_moves(player)

        # 2. Если есть ходы со взятием, мы обязаны сделать один из них
        if capture_moves:
            return capture_moves

        # 3. Если ходов со взятием нет, ищем обычные ходы
        simple_moves = self._find_all_simple_moves(player)
        return simple_moves

    def make_move(self, move):
        """
        Применяет ход на доске (улучшенная версия для дамок).
        """
        start_pos = move[0]
        end_pos = move[-1]
        piece = self.board[start_pos]
        self.board[start_pos] = 0

        # Определяем, был ли это ход со взятием
        is_capture = False
        if len(move) > 1:
            # Если разница по вертикали/горизонтали больше 1, это точно взятие
            if abs(move[0][0] - move[1][0]) > 1:
                is_capture = True

        if is_capture:
            # Проходим по каждому "прыжку" в цепочке
            for i in range(len(move) - 1):
                pos1 = move[i]
                pos2 = move[i + 1]

                # Сканируем диагональ между pos1 и pos2, чтобы найти сбитую шашку
                dr = np.sign(pos2[0] - pos1[0])
                dc = np.sign(pos2[1] - pos1[1])

                scan_r, scan_c = pos1[0] + dr, pos1[1] + dc
                while (scan_r, scan_c) != pos2:
                    if self.board[scan_r, scan_c] != 0:
                        self.board[scan_r, scan_c] = 0  # Нашли и удалили
                        break
                    scan_r += dr
                    scan_c += dc

        self.board[end_pos] = piece

        # Проверка на превращение в дамку
        if piece == 1 and end_pos[0] == 0:
            self.board[end_pos] = 2
        elif piece == -1 and end_pos[0] == config.BOARD_Y - 1:
            self.board[end_pos] = -2

    def get_canonical_form(self, board, player):
        """
        Приводит доску к виду от первого лица.
        Сеть всегда должна видеть доску со своей стороны.
        """
        return board * player

    def _find_all_simple_moves(self, player):
        """ Находит все 'тихие' ходы (без взятия) для игрока. """
        moves = []
        # Направление движения пешек (для белых -1 (вверх), для черных 1 (вниз))
        move_dir = -1 if player == 1 else 1

        for r in range(config.BOARD_Y):
            for c in range(config.BOARD_X):
                piece = self.board[r, c]

                # --- Ходы пешек ---
                if piece == player:
                    # Проверяем два диагональных хода вперед
                    for dc in [-1, 1]:  # Изменение по колонке: -1 (влево), 1 (вправо)
                        nr, nc = r + move_dir, c + dc

                        if self.is_valid_pos(nr, nc) and self.board[nr, nc] == 0:
                            moves.append([(r, c), (nr, nc)])

                # --- Ходы дамок ---
                elif piece == player * 2:
                    # Проверяем все 4 диагональных направления
                    for dr in [-1, 1]:
                        for dc in [-1, 1]:
                            # Двигаемся по диагонали шаг за шагом
                            for step in range(1, config.BOARD_X):
                                nr, nc = r + dr * step, c + dc * step

                                if self.is_valid_pos(nr, nc):
                                    if self.board[nr, nc] == 0:
                                        moves.append([(r, c), (nr, nc)])
                                    else:
                                        # Уперлись в фигуру, дальше в этом направлении идти нельзя
                                        break
                                else:
                                    # Вышли за пределы доски
                                    break
        return moves

    def _find_all_capture_moves(self, player):
        """ Находит все ходы со взятием для игрока (для пешек и дамок). """
        all_capture_sequences = []
        for r in range(config.BOARD_Y):
            for c in range(config.BOARD_X):
                piece = self.board[r, c]
                # --- Поиск для пешек ---
                if piece == player:
                    # Убираем лишние аргументы, они теперь по умолчанию
                    sequences = self._find_pawn_captures_recursive(self.board, r, c, player, [(r, c)], [])
                    all_capture_sequences.extend(sequences)

                elif piece == player * 2:
                    # Убираем лишние аргументы, они теперь по умолчанию
                    sequences = self._find_king_captures_recursive(self.board, r, c, player, [(r, c)], [])
                    all_capture_sequences.extend(sequences)

        if not all_capture_sequences:
            return []

        max_len = max(len(seq) for seq in all_capture_sequences)
        longest_moves = [seq for seq in all_capture_sequences if len(seq) == max_len]

        return longest_moves

    def _find_pawn_captures_recursive(self, board, r, c, player, current_path, captured_in_path, depth=0):
        """ Рекурсивно ищет цепочки взятий для ПЕШКИ с защитой от циклов. """

        # --- СТРАХОВОЧНЫЙ ТРОС ---
        if depth > 20:  # Максимальная глубина серии взятий
            return []

        found_sequences = []
        for dr in [-1, 1]:
            for dc in [-1, 1]:
                mid_r, mid_c = r + dr, c + dc
                end_r, end_c = r + 2 * dr, c + 2 * dc

                if self.is_valid_pos(end_r, end_c) and board[end_r, end_c] == 0:
                    enemy_piece = board[mid_r, mid_c]
                    is_opponent = enemy_piece != 0 and np.sign(enemy_piece) != np.sign(player)

                    if is_opponent and (mid_r, mid_c) not in captured_in_path:
                        new_path = current_path + [(end_r, end_c)]
                        new_captured = captured_in_path + [(mid_r, mid_c)]

                        next_board = board.copy()
                        next_board[r, c] = 0
                        next_board[mid_r, mid_c] = 0
                        next_board[end_r, end_c] = player

                        # Передаем увеличенную глубину
                        further_sequences = self._find_pawn_captures_recursive(next_board, end_r, end_c, player,
                                                                               new_path, new_captured, depth + 1)

                        if further_sequences:
                            found_sequences.extend(further_sequences)
                        else:
                            found_sequences.append(new_path)
        return found_sequences

    def _find_king_captures_recursive(self, board, r, c, player, current_path, captured_in_path, depth=0):
        """ Рекурсивно ищет цепочки взятий для ДАМКИ (финальная, защищенная версия). """

        if depth > 20:  # Максимальная разумная длина серии взятий
            return []

        found_sequences = []

        # Проверяем все 4 диагональных направления
        for dr in [-1, 1]:
            for dc in [-1, 1]:
                enemy_pos = None
                # Ищем первую фигуру на диагонали
                for step in range(1, config.BOARD_X):
                    scan_r, scan_c = r + dr * step, c + dc * step

                    if not self.is_valid_pos(scan_r, scan_c):
                        break

                    # Работаем только с переданной доской 'board'
                    piece_on_path = board[scan_r, scan_c]

                    # Если наткнулись на свою фигуру, диагональ заблокирована
                    if piece_on_path != 0 and np.sign(piece_on_path) == np.sign(player):
                        break

                    # Если нашли фигуру противника, которую еще не били в этой серии
                    is_opponent = piece_on_path != 0 and np.sign(piece_on_path) != np.sign(player)
                    if is_opponent and (scan_r, scan_c) not in captured_in_path:
                        enemy_pos = (scan_r, scan_c)
                        break

                # Если нашли врага, ищем все пустые клетки за ним
                if enemy_pos:
                    enemy_r, enemy_c = enemy_pos
                    for step in range(1, config.BOARD_X):
                        land_r, land_c = enemy_r + dr * step, enemy_c + dc * step

                        # Если вышли за пределы доски или клетка не пуста, дальше в этом направлении не ищем
                        if not self.is_valid_pos(land_r, land_c) or board[land_r, land_c] != 0:
                            break

                        # Защита от циклов, когда дамка возвращается на пройденное поле
                        if (land_r, land_c) in current_path:
                            continue

                        new_path = current_path + [(land_r, land_c)]
                        new_captured = captured_in_path + [enemy_pos]

                        # Создаем НОВУЮ гипотетическую доску для следующего шага рекурсии
                        next_board = board.copy()
                        next_board[r, c] = 0  # Убираем дамку со старого места
                        next_board[enemy_pos] = 0  # Убираем сбитую шашку
                        next_board[land_r, land_c] = player * 2  # Ставим дамку на новое место

                        # Рекурсивный вызов с новой доской и увеличенной глубиной
                        further_sequences = self._find_king_captures_recursive(next_board, land_r, land_c, player,
                                                                               new_path, new_captured, depth + 1)

                        if further_sequences:
                            found_sequences.extend(further_sequences)
                        else:
                            found_sequences.append(new_path)

        return found_sequences

    def check_game_over(self, player):
        """
        Проверяет, закончилась ли игра для игрока player, который должен ходить.
        Возвращает:
        1 - если игрок 'player' выиграл
        -1 - если игрок 'player' проиграл
        0 - если игра продолжается
        """
        # 1. Проверяем, есть ли у противника фигуры
        opponent = -player
        opponent_pieces = np.count_nonzero(
            (self.board == opponent) | (self.board == opponent * 2)
        )
        if opponent_pieces == 0:
            return 1  # Игрок 'player' выиграл, у врага нет фигур

        # 2. Проверяем, есть ли у игрока 'player' ходы
        possible_moves = self.get_valid_moves(player)
        if not possible_moves:
            return -1  # Игрок 'player' проиграл, ему некуда ходить

        # Если ни одно из условий не выполнено, игра продолжается
        return 0


    def getActionSize(self):
        """Возвращает количество всех возможных ходов (размер вектора политики)."""
        # Мы используем простое кодирование "откуда-куда", поэтому размер
        # вектора равен 8*8 * 8*8, что соответствует config.ACTION_SIZE
        return config.ACTION_SIZE

    def move_to_action(self, move):
        """
        Превращает ход в уникальное число.
        Кодирует все 4 координаты: y1, x1, y2, x2.
        """
        start_pos, end_pos = move[0], move[-1]
        y1, x1 = start_pos
        y2, x2 = end_pos
        # Уникальное кодирование для доски 8x8
        return (y1 * 8 + x1) * 64 + (y2 * 8 + x2)

    def action_to_move(self, action):
        # Эта функция теперь не нужна MCTS, но оставим заглушку
        # Реальная логика восстановления хода теперь в get_next_state
        return None

    def get_string_representation(self, board):
        """
        Возвращает уникальное строковое (байтовое) представление доски.
        Нужно для использования в качестве ключа в словарях MCTS.
        """
        return board.tobytes()

    def get_next_state(self, board, player, action):
        """
        Возвращает следующее состояние доски после выполнения действия 'action'.
        Эта версия надежно находит полный ход 'move' по его 'action'.
        """
        # 1. Находим все возможные ходы из текущей позиции
        valid_moves = self.get_valid_moves_for_board(board, player)

        # 2. Ищем тот самый ход, который соответствует выбранному действию
        move_to_execute = None
        for move in valid_moves:
            if self.move_to_action(move) == action:
                move_to_execute = move
                break

        # 3. "Страховка": если по какой-то причине ход не найден,
        # (например, из-за ошибки в MCTS), не падаем, а просто возвращаем
        # исходное состояние, чтобы игра могла продолжаться.
        if move_to_execute is None:
            # Можно добавить print, чтобы знать о таких случаях
            # print(f"Внимание: не удалось найти ход для действия {action}")
            return (board, -player)

        # 4. Безопасно выполняем найденный ход на временной копии
        b = np.copy(board)
        temp_game = CheckersGame()
        temp_game.board = b
        temp_game.make_move(move_to_execute)  # Выполняем ПОЛНЫЙ, правильный ход

        return (temp_game.board, -player)


    def get_valid_moves_for_board(self, board, player):
        """ Версия get_valid_moves, которая работает с предоставленной доской. """
        # Временно подменяем доску объекта, чтобы использовать существующую логику
        original_board = self.board
        self.board = board
        moves = self.get_valid_moves(player)
        # Возвращаем доску в исходное состояние
        self.board = original_board
        return moves

