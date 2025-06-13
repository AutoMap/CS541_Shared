
# other_checkers_players.py
"""
A handful of non-learning opponents:

* MinimaxAlphaBetaCheckersPlayer – depth-limited search with α-β pruning
* RandomCheckersPlayer           – picks a random legal move
* HumanCheckersPlayer            – text-console input (GUI hooks can be added)
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from draughts import Board, Move

from checkers_player_interface import CheckersPlayerConfiguration, AbstractCheckersPlayer


# ────────────────────────── common helpers ─────────────────────────────
def _sq32_to_rc(index32: int) -> Tuple[int, int]:
    """32-square index → (row,col) on 8×8 board."""
    row = index32 // 4
    col = (index32 % 4) * 2 + ((row + 1) & 1)
    return row, col


# ──────────────────────── minimax with α-β ─────────────────────────────
class MinimaxAlphaBetaCheckersPlayer(AbstractCheckersPlayer):
    def __init__(self, cfg: CheckersPlayerConfiguration):
        super().__init__(cfg)
        p = cfg.configuration_parameters
        self.depth = p.get("search_depth", 4)

    # main interface -----------------------------------------------------
    def choose_move_from_legal_actions(
        self, current_board_state: np.ndarray, available_legal_actions: List[int]
    ) -> int:
        board = self._to_draughts_board(current_board_state)
        legal_moves = list(board.legal_moves())
        if len(legal_moves) == 1:
            return 0

        best_score = float("-inf")
        best_idx = 0
        for idx, move in enumerate(legal_moves):
            board.push(move)
            score = -self._negamax(board, self.depth - 1, float("-inf"), float("inf"))
            board.pop()
            if score > best_score:
                best_score = score
                best_idx = idx
        return available_legal_actions[best_idx]

    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        self.record_game_outcome(
            did_win=game_result_data.get("did_win", False),
            was_draw=game_result_data.get("was_draw", False),
            additional_info=game_result_data,
        )

    # inner search -------------------------------------------------------
    def _negamax(self, board: Board, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or board.is_over():
            return self._evaluate(board)

        best = float("-inf")
        for mv in board.legal_moves():
            board.push(mv)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    # very simple evaluation (material only) ----------------------------
    @staticmethod
    def _evaluate(board: Board) -> float:
        score = 0
        for p in board.board:
            if p == 1:     # white man
                score += 1
            elif p == 3:   # white king
                score += 1.5
            elif p == -1:  # black man
                score -= 1
            elif p == -3:  # black king
                score -= 1.5
        return score if board.whose_turn() == 1 else -score

    # -------------------------------------------------------------------
    @staticmethod
    def _to_draughts_board(state: np.ndarray) -> Board:
        """Convert 8×8 numpy view back to a Board for search."""
        board = Board(variant="english", fen="startpos")
        board.clear()
        for sq in range(32):
            r, c = _sq32_to_rc(sq)
            v = state[r, c]
            if v == 1:
                board.board[sq] = 1
            elif v == 2:
                board.board[sq] = 3
            elif v == -1:
                board.board[sq] = -1
            elif v == -2:
                board.board[sq] = -3
        board.turn = 1  # perspective is always current player
        return board


# ─────────────────────────── random player ─────────────────────────────
class RandomCheckersPlayer(AbstractCheckersPlayer):
    def __init__(self, cfg: CheckersPlayerConfiguration):
        super().__init__(cfg)
        seed = cfg.configuration_parameters.get("random_seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def choose_move_from_legal_actions(
        self, current_board_state: np.ndarray, available_legal_actions: List[int]
    ) -> int:
        return random.choice(available_legal_actions)

    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        self.record_game_outcome(
            did_win=game_result_data.get("did_win", False),
            was_draw=game_result_data.get("was_draw", False),
            additional_info=game_result_data,
        )


# ─────────────────────────── human player ──────────────────────────────
class HumanCheckersPlayer(AbstractCheckersPlayer):
    def choose_move_from_legal_actions(
        self, current_board_state: np.ndarray, available_legal_actions: List[int]
    ) -> int:
        print("\nCurrent board:")
        print(current_board_state)
        print("Legal actions:", available_legal_actions)
        while True:
            try:
                idx = int(input("Pick action index: "))
                if idx in available_legal_actions:
                    return idx
            except ValueError:
                pass
            print("Invalid input; try again.")

    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        self.record_game_outcome(
            did_win=game_result_data.get("did_win", False),
            was_draw=game_result_data.get("was_draw", False),
            additional_info=game_result_data,
        )


# ────────────────────────── factory helper ─────────────────────────────
def create_player_from_configuration(
    player_type: str,
    configuration_parameters: Dict[str, Any],
    model_file_path: str | None = None,
) -> AbstractCheckersPlayer:
    cfg = CheckersPlayerConfiguration(
        player_type_name=player_type,
        configuration_parameters=configuration_parameters,
        model_file_path=model_file_path,
    )
    if player_type == "minimax":
        return MinimaxAlphaBetaCheckersPlayer(cfg)
    if player_type == "random":
        return RandomCheckersPlayer(cfg)
    if player_type == "human":
        return HumanCheckersPlayer(cfg)
    raise ValueError(f"Unknown player type '{player_type}'")
