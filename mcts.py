# Алгоритм поиска по дереву Монте-Карло
# mcts.py

import math
import numpy as np
import config
from utils import get_nnet_input_tensor


class MCTS:

    # Этот класс отвечает за логику поиска по дереву Монте-Карло.

    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet

        # Словари для хранения статистики дерева
        self.Qsa = {}  # Хранит Q-значения для пар (насколько хорош ход a в позиции s)
        self.Nsa = {}  # Хранит количество посещений для пар (столько раз, сколько мы выбрали ход a в позиции s)
        self.Ns = {}  # Хранит количество посещений для состояния (сколько раз мы были в позиции s)
        self.Ps = {}  # Хранит начальные вероятности (предсказания нейросети)
        self.device = next(nnet.parameters()).device

    def getActionProb(self, canonicalBoard, turn=0, temp=1):
        """
        Главная функция. Выполняет симуляции и возвращает вектор вероятностей.
        Принимает номер хода 'turn' для управления шумом.
        """
        for _ in range(config.MCTS_SIMULATIONS):
            # Передаем 'turn' в search
            self.search(canonicalBoard, turn=turn)

        s = self.game.get_string_representation(canonicalBoard)
        valid_moves = self.game.get_valid_moves_for_board(canonicalBoard, 1)
        counts = {self.game.move_to_action(m): self.Nsa.get((s, self.game.move_to_action(m)), 0) for m in valid_moves}

        if not counts:
            return np.ones(self.game.getActionSize()) / self.game.getActionSize()

        probs = np.zeros(self.game.getActionSize())

        if temp == 0:
            best_action = max(counts, key=counts.get)
            probs[best_action] = 1
            return probs

        total_visits = sum(counts.values())
        if total_visits == 0:  # Доп. страховка от деления на ноль
            return np.ones(self.game.getActionSize()) / self.game.getActionSize()

        for action, count in counts.items():
            probs[action] = count / total_visits

        return probs

    def search(self, canonicalBoard_np, turn=0, depth=0, visited=None):
        """
        Рекурсивная функция поиска с защитой от циклов позиций.
        """
        # Убираем MAX_SEARCH_DEPTH, он нам больше не нужен
        # if depth > MAX_SEARCH_DEPTH: return 0

        # Инициализируем историю для самого первого вызова
        if visited is None:
            visited = set()

        s = self.game.get_string_representation(canonicalBoard_np)

        # Если мы уже были в этой позиции в этой ветке симуляции, это цикл.
        # Засчитываем ничью и не углубляемся дальше.
        if s in visited:
            return 0

        # Добавляем текущую позицию в историю этой ветки
        visited.add(s)

        # конец рекурсии: узел является терминальным
        valid_moves_for_check = self.game.get_valid_moves_for_board(canonicalBoard_np, 1)
        if not valid_moves_for_check:
            return 1  # Выигрыш для родителя

        opponent_pieces = np.count_nonzero((canonicalBoard_np == -1) | (canonicalBoard_np == -2))
        if opponent_pieces == 0:
            return -1  # Проигрыш для родителя

        # EXPANSION (РАСШИРЕНИЕ)
        if s not in self.Ps:
            board_tensor = get_nnet_input_tensor(canonicalBoard_np, 1).to(self.device)
            policy, value = self.nnet.predict(board_tensor)

            # УМНЫЙ ШУМ
            # Добавляем шум только в корневом узле (depth=0) и только в начале игры (например, первые 10 ходов)
            if depth == 0 and turn < 10:
                epsilon = 0.25  # Можно вернуть 25% шума, т.к. он будет применяться редко

                legal_actions = [self.game.move_to_action(m) for m in valid_moves_for_check]
                noise = np.random.dirichlet([0.3] * len(legal_actions))

                noise_mask = np.zeros_like(policy)
                for i, action_idx in enumerate(legal_actions):
                    if action_idx < len(noise_mask):
                        noise_mask[action_idx] = noise[i]

                self.Ps[s] = (1 - epsilon) * policy + epsilon * noise_mask
            else:
                # В середине игры или в глубине дерева шум не добавляем
                self.Ps[s] = policy

            self.Ns[s] = 0
            return -value

        # SELECTION (ВЫБОР) 
        # Мы уже получили valid_moves для проверки выше, используем их
        valid_moves = valid_moves_for_check
        if not valid_moves:
            return 0  # Ничья, если нет ходов, но игра не окончена

        best_uct = -float('inf')
        # Выбираем первый ход как лучший по умолчанию, чтобы избежать ситуации, когда ничего не выбрано
        best_act = self.game.move_to_action(valid_moves[0])

        for move in valid_moves:
            a = self.game.move_to_action(move)
            if (s, a) in self.Qsa:
                uct = self.Qsa[(s, a)] + config.CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                uct = config.CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

            if uct > best_uct:
                best_uct = uct
                best_act = a

        a = best_act
        move_to_do = None
        for m in valid_moves:
            if self.game.move_to_action(m) == a:
                move_to_do = m
                break

        next_board_np, next_player = self.game.get_next_state(canonicalBoard_np, 1, move_to_do)
        next_board_np = self.game.get_canonical_form(next_board_np, next_player)

        # Передаем историю посещенных узлов дальше
        value = self.search(next_board_np, turn=turn, depth=depth + 1, visited=visited)

        # BACKPROPAGATION (ОБРАТНОЕ РАСПРОСТРАНЕНИЕ)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + value) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = value
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -value
