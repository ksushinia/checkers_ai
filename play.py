# interface.py

import pygame
import numpy as np
import torch
import sys
import os

from game import CheckersGame
from network import CheckersNet
from mcts import MCTS

# --- НАСТРОЙКИ ИГРЫ И ИНТЕРФЕЙСА ---
CHECKPOINT_FILE = 'tars/Exp2/little/Exp2-400-lite.tar'  # Имя файла с обученной моделью
HUMAN_PLAYER = -1  # За кого ты будешь играть: -1 за черных (ходят вторыми), 1 за белых (ходят первыми)

# Настройки графики
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 640
BOARD_ROWS, BOARD_COLS = 8, 8
SQUARE_SIZE = SCREEN_WIDTH // BOARD_COLS

# Цвета
COLOR_WHITE_SQUARE = (238, 238, 210)
COLOR_BLACK_SQUARE = (118, 150, 86)
COLOR_PLAYER_1 = (240, 240, 240)  # Белые шашки
COLOR_PLAYER_2 = (50, 50, 50)  # Черные шашки
COLOR_KING = (255, 215, 0)  # Золотой цвет для коронки дамок
COLOR_HIGHLIGHT = (186, 202, 68, 150)  # Полупрозрачный желтый для выделения
COLOR_VALID_MOVE = (100, 150, 255, 150)  # Полупрозрачный синий для возможных ходов


# --- КЛАСС ДЛЯ УПРАВЛЕНИЯ AI ---
class CheckersAI:
    def __init__(self, checkpoint_path, game):
        self.game = game
        self.device = torch.device("cpu")  # Для игры достаточно CPU
        self.nnet = CheckersNet().to(self.device)

        try:
            # Загружаем обученную модель
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.nnet.load_state_dict(checkpoint['state_dict'])
            self.nnet.eval()
            print("Модель AI успешно загружена.")
        except FileNotFoundError:
            print(f"Ошибка: Файл чекпоинта '{checkpoint_path}' не найден. AI не будет работать.")
            self.nnet = None
        except Exception as e:
            print(f"Произошла ошибка при загрузке модели: {e}")
            self.nnet = None

    def get_best_move(self, board, player):
        if not self.nnet:
            return None

        # AI "думает" над ходом
        canonical_board = self.game.get_canonical_form(board, player)
        mcts = MCTS(self.game, self.nnet)
        pi = mcts.getActionProb(canonical_board, temp=0)  # temp=0 для выбора лучшего хода
        action = np.argmax(pi)

        # Находим реальный ход, соответствующий лучшему действию
        valid_moves = self.game.get_valid_moves_for_board(board, player)
        for move in valid_moves:
            if self.game.move_to_action(move) == action:
                return move

        # Если лучший ход по какой-то причине не найден, возвращаем любой случайный
        return valid_moves[0] if valid_moves else None


# --- ФУНКЦИИ ОТРИСОВКИ ---
def draw_board(screen):
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            color = COLOR_WHITE_SQUARE if (r + c) % 2 == 0 else COLOR_BLACK_SQUARE
            pygame.draw.rect(screen, color, (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(screen, board, piece_to_skip=None):
    radius = SQUARE_SIZE // 2 - 8
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            # Пропускаем отрисовку движущейся шашки
            if (r, c) == piece_to_skip:
                continue

            piece = board[r][c]
            if piece != 0:
                color = COLOR_PLAYER_1 if np.sign(piece) == 1 else COLOR_PLAYER_2
                center = (c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2)
                pygame.draw.circle(screen, color, center, radius)
                # Если это дамка, рисуем "корону"
                if abs(piece) == 2:
                    pygame.draw.circle(screen, COLOR_KING, center, radius // 2)


def draw_highlight(screen, position):
    if position:
        r, c = position
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        surface.fill(COLOR_HIGHLIGHT)
        screen.blit(surface, (c * SQUARE_SIZE, r * SQUARE_SIZE))


def draw_valid_moves(screen, moves):
    for r, c in moves:
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        center = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
        pygame.draw.circle(surface, COLOR_VALID_MOVE, center, 15)
        screen.blit(surface, (c * SQUARE_SIZE, r * SQUARE_SIZE))


def animate_move(screen, board_before, move, piece_type, clock):
    """Анимирует перемещение одной шашки."""

    # Длительность анимации в миллисекундах
    ANIMATION_SPEED = 200  # 0.2 секунды

    # Определяем цвет и тип шашки
    color = COLOR_PLAYER_1 if np.sign(piece_type) == 1 else COLOR_PLAYER_2
    is_king = abs(piece_type) == 2
    radius = SQUARE_SIZE // 2 - 8

    # Проходим по каждому "прыжку" в ходе (для серий взятий)
    for i in range(len(move) - 1):
        start_pos, end_pos = move[i], move[i + 1]
        start_r, start_c = start_pos
        end_r, end_c = end_pos

        # Рассчитываем начальные и конечные пиксельные координаты
        start_pixel_x = start_c * SQUARE_SIZE + SQUARE_SIZE // 2
        start_pixel_y = start_r * SQUARE_SIZE + SQUARE_SIZE // 2
        end_pixel_x = end_c * SQUARE_SIZE + SQUARE_SIZE // 2
        end_pixel_y = end_r * SQUARE_SIZE + SQUARE_SIZE // 2

        start_time = pygame.time.get_ticks()

        # Цикл самой анимации для одного прыжка
        while True:
            elapsed = pygame.time.get_ticks() - start_time

            # Процент завершения анимации
            progress = min(elapsed / ANIMATION_SPEED, 1.0)

            # Интерполируем текущие координаты
            current_x = start_pixel_x + (end_pixel_x - start_pixel_x) * progress
            current_y = start_pixel_y + (end_pixel_y - start_pixel_y) * progress

            # --- Отрисовка кадра анимации ---
            # 1. Рисуем доску
            draw_board(screen)
            # 2. Рисуем ВСЕ шашки, КРОМЕ движущейся
            temp_board = board_before.copy()
            temp_board[start_r][start_c] = 0  # Скрываем движущуюся шашку с начальной позиции
            draw_pieces(screen, temp_board)
            # 3. Рисуем движущуюся шашку в ее текущем положении
            pygame.draw.circle(screen, color, (current_x, current_y), radius)
            if is_king:
                pygame.draw.circle(screen, COLOR_KING, (current_x, current_y), radius // 2)

            pygame.display.flip()
            # --- Конец отрисовки кадра ---

            if progress >= 1.0:
                break

            clock.tick(60)  # Поддерживаем 60 FPS

        # Обновляем доску для следующего прыжка (если это серия взятий)
        board_before[end_r][end_c] = piece_type
        # Удаляем сбитую шашку
        if abs(start_r - end_r) > 1:
            mid_r, mid_c = (start_r + end_r) // 2, (start_c + end_c) // 2
            board_before[mid_r][mid_c] = 0

# --- ОСНОВНАЯ ЛОГИКА ИГРЫ ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Робот-шашист")
    clock = pygame.time.Clock()

    game = CheckersGame()
    ai = CheckersAI(CHECKPOINT_FILE, game)

    board = game.get_board_state()
    current_player = 1
    selected_piece = None
    valid_destinations = {}

    # --- НОВЫЕ ПЕРЕМЕННЫЕ ДЛЯ АНИМАЦИИ ---
    animating_move = None  # Хранит ход, который анимируется в данный момент
    board_before_animation = None

    running = True
    while running:
        game_over = game.check_game_over(current_player)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Обрабатываем клики только если нет анимации
            if not animating_move and game_over == 0 and current_player == HUMAN_PLAYER and event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                c, r = pos[0] // SQUARE_SIZE, pos[1] // SQUARE_SIZE

                if selected_piece:
                    if (r, c) in valid_destinations:
                        # --- ИЗМЕНЕНИЕ: Запускаем анимацию вместо мгновенного хода ---
                        animating_move = valid_destinations[(r, c)]
                        board_before_animation = board.copy()
                    selected_piece = None
                    valid_destinations = {}
                else:
                    if np.sign(board[r][c]) == HUMAN_PLAYER:
                        selected_piece = (r, c)
                        all_valid_moves = game.get_valid_moves_for_board(board, HUMAN_PLAYER)
                        piece_moves = [m for m in all_valid_moves if m[0] == selected_piece]
                        valid_destinations = {m[-1]: m for m in piece_moves}

        # --- НОВЫЙ БЛОК: Запускаем ход AI, только если нет анимации ---
        if not animating_move and game_over == 0 and current_player != HUMAN_PLAYER:
            pygame.display.set_caption("Робот-шашист (AI думает...)")
            # Перерисовываем экран, чтобы надпись "AI думает" появилась сразу
            draw_board(screen)
            draw_pieces(screen, board)
            pygame.display.flip()

            move = ai.get_best_move(board, current_player)
            if move:
                animating_move = move
                board_before_animation = board.copy()
            else:
                game_over = -current_player
            pygame.display.set_caption("Робот-шашист")

        # --- ОСНОВНАЯ ОТРИСОВКА ---
        draw_board(screen)
        draw_highlight(screen, selected_piece)
        draw_valid_moves(screen, valid_destinations.keys())
        draw_pieces(screen, board)

        # --- НОВЫЙ БЛОК: Выполняем анимацию, если она есть ---
        if animating_move:
            start_pos = animating_move[0]
            piece_type = board_before_animation[start_pos]

            # Запускаем саму анимацию. Основная доска 'board' не меняется.
            animate_move(screen, board_before_animation, animating_move, piece_type, clock)

            # После анимации обновляем состояние игры
            # Мы передаем СТАРУЮ доску в get_next_state, чтобы он правильно рассчитал новый ход
            board, current_player = game.get_next_state(board, current_player, game.move_to_action(animating_move))

            # Завершаем анимацию
            animating_move = None

        if game_over != 0:
            font = pygame.font.SysFont('Arial', 50)
            winner = "Вы победили!" if game_over == HUMAN_PLAYER else "AI Победил!"
            text = font.render(winner, True, (255, 0, 0))
            text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()