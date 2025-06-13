#!/usr/bin/env python3
"""
English‑draughts (8 × 8) environment based on **pydraughts 0.6.7**.

Changes in this version
───────────────────────
• All public state‑returning methods now deliver **(board, turn)** where
  `turn` is the actual colour that will move next: `'W'` or `'B'`.
• Board orientation is still from the mover’s viewpoint (positive values
  are always the player who is about to move).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from draughts import Board


# ───────────────────────── data class ─────────────────────────
@dataclass
class CheckersGameResult:
    winning_player_name: str          # "player_one" | "player_two" | "draw"
    total_game_moves: int
    final_pieces_count: Dict[str, int]
    game_duration_seconds: float
    player_one_type: str
    player_two_type: str
    game_timestamp: str
    player_one_configuration: Optional[Dict] = None
    player_two_configuration: Optional[Dict] = None


# ─────────────────────── environment class ───────────────────
class CheckersGameEnvironment:
    DARK_SQUARES = 32  # English variant uses 32 playable squares

    def __init__(self, enable_detailed_logging: bool = False):
        self.enable_detailed_logging = enable_detailed_logging
        self.move_history_list: List[str] = []
        self.reset_game_to_initial_state()

    # public ---------------------------------------------------
    def reset_game_to_initial_state(self) -> Tuple[np.ndarray, str]:
        """Create a fresh board and return **(board, side_to_move)**."""
        self.pydraughts_board: Board = Board(variant="english", fen="startpos")
        self.is_game_finished = False
        self.winning_player_name: Optional[str] = None
        self.total_moves_played = 0
        self.game_start_timestamp = time.time()
        self.move_history_list.clear()
        return self.get_current_board_state()

    def get_current_board_state(self) -> Tuple[np.ndarray, str]:
        """
        Returns
        -------
        board : 8 × 8 np.ndarray
            Encoded from the *mover’s* perspective.
        turn  : str
            `'W'` if White is to play, `'B'` if Black is to play.
        """
        board = np.zeros((8, 8), dtype=int)

        # FEN looks like "B:W18,22,25,K10:B3,11,15"
        fen = self.pydraughts_board.fen
        side_to_move, white_part, black_part = fen.split(":")

        def _fill(part: str, is_white: bool):
            if len(part) <= 1:
                return
            for token in part[1:].split(","):
                is_king = token.startswith("K")
                square_index = int(token[1:] if is_king else token) - 1  # 1‑based → 0‑based
                r, c = self._sq32_to_rc(square_index)
                val = 2 if is_king else 1
                if not is_white:
                    val = -val
                board[r, c] = val

        _fill(white_part, True)
        _fill(black_part, False)

        # Flip signs so that “positive = side_to_move”
        if side_to_move == "B":
            board = -board
        return board, side_to_move

    def get_all_legal_action_indices(self) -> List[int]:
        return list(range(len(self.pydraughts_board.legal_moves())))

    def execute_action_by_index(self, action_idx: int):
        """
        Returns
        -------
        (board, turn) : Tuple[np.ndarray, str]  – state after the move
        reward        : float
        done          : bool
        info          : dict
        """
        move = list(self.pydraughts_board.legal_moves())[action_idx]
        self.pydraughts_board.push(move)
        self.total_moves_played += 1

        reward = self._calc_reward()
        self.is_game_finished = self.pydraughts_board.is_over()
        if self.is_game_finished:
            self.winning_player_name = self._determine_winner()

        info = {"move": str(move), "ply": self.total_moves_played}
        return self.get_current_board_state(), reward, self.is_game_finished, info

    # helpers --------------------------------------------------
    @staticmethod
    def _sq32_to_rc(i: int) -> Tuple[int, int]:
        row = i // 4
        col = (i % 4) * 2 + ((row + 1) & 1)
        return row, col

    def _calc_reward(self) -> float:
        if not self.pydraughts_board.is_over():
            return 0.01
        winner = self.pydraughts_board.winner()  # 2=WHITE, 1=BLACK, 0=draw
        if winner == 2:
            return 1.0
        if winner == 1:
            return -1.0
        return 0.0

    def _determine_winner(self) -> Optional[str]:
        if not self.is_game_finished:
            return None
        return {2: "player_one", 1: "player_two", 0: "draw"}.get(
            self.pydraughts_board.winner(), "draw"
        )
