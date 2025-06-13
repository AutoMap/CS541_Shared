"""
Q‑learning agent for 8 × 8 English draughts.

Key change
──────────
The *colour to move* is now embedded in every Q‑state key so positions
with identical material but different players to act are learnt
separately.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from checkers_player_interface import (
    AbstractTrainableCheckersPlayer,
    CheckersPlayerConfiguration,
)

# ───────────────────── helpers ──────────────────────
def _sq32_to_rc(i: int) -> tuple[int, int]:
    return i // 4, (i % 4) * 2 + ((i // 4 + 1) & 1)


def _state_to_key(board: np.ndarray, turn: str) -> str:
    """
    Build a *stable* textual key that starts with the actual side to move.

    Example:  'B:W18,22:K25:B3,11'  (arbitrary ordering is fine for learning)
    """
    w_m, w_k, b_m, b_k = [], [], [], []
    for sq in range(32):
        r, c = _sq32_to_rc(sq)
        v = board[r, c]
        if v == 1:
            w_m.append(str(sq + 1))
        elif v == 2:
            w_k.append("K" + str(sq + 1))
        elif v == -1:
            b_m.append(str(sq + 1))
        elif v == -2:
            b_k.append("K" + str(sq + 1))
    return f"{turn}:W{','.join(w_k + w_m)}:B{','.join(b_k + b_m)}"


def _coerce_param(d: Dict[str, Any], *names, default):
    for n in names:
        if n in d:
            return d[n]
    return default


# ───────────────────── learner ──────────────────────
class QLearningCheckersPlayer(AbstractTrainableCheckersPlayer):
    def __init__(self, cfg: CheckersPlayerConfiguration):
        super().__init__(cfg)
        p = cfg.configuration_parameters

        self.alpha       = float(_coerce_param(p, "lr", "learning_rate", "alpha", default=0.2))
        self.gamma       = float(_coerce_param(p, "gamma", "discount_factor", default=0.95))
        self.epsilon     = float(_coerce_param(p, "epsilon", "initial_epsilon", default=0.8))
        self.eps_decay   = float(p.get("epsilon_decay", 0.02))
        self.decay_every = int(p.get("decay_interval", 1))
        self.save_path: str | None = p.get("save_path")

        self.q: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        if cfg.model_file_path:
            self.load_trained_model_from_file(cfg.model_file_path)

        self.games_played = 0

    # ---------- core required API ----------
    def choose_move_from_legal_actions(
        self, current_board_state: np.ndarray, available_legal_actions: List[int]
    ) -> int:
        # _current_turn is injected by checkers_main without changing the interface
        turn = getattr(self, "_current_turn", "W")  # old tables default to 'W'

        key = _state_to_key(current_board_state, turn)
        _ = self.q[key]  # ensure row present

        if random.random() < self.epsilon:
            return random.choice(available_legal_actions)

        qs = self.q[key]
        return max(available_legal_actions, key=lambda a: qs.get(a, 0.0))

    # ---------- learning step ----------
    def learn_one_transition(
        self,
        prev_board: np.ndarray,
        prev_turn: str,
        action_taken: int,
        reward: float,
        next_board: np.ndarray,
        next_turn: str,
        next_legal_actions: List[int],
    ):
        s1 = _state_to_key(prev_board, prev_turn)
        s2 = _state_to_key(next_board, next_turn)
        _ = self.q[s2]  # auto‑init next row

        best_next = max((self.q[s2].get(a, 0.0) for a in next_legal_actions), default=0.0)
        old = self.q[s1].get(action_taken, 0.0)
        self.q[s1][action_taken] = old + self.alpha * (reward + self.gamma * best_next - old)

    # wrappers expected by AbstractTrainableCheckersPlayer
    def update_from_game_experience(self, *args, **kwargs):  # type: ignore[override]
        self.learn_one_transition(*args, **kwargs)

    def update_after_game_completion(self, _):  # type: ignore[override]
        self.games_played += 1
        if self.games_played % self.decay_every == 0:
            self.epsilon = max(0.0, self.epsilon - self.eps_decay)

    def get_training_progress_statistics(self) -> Dict[str, Any]:
        return {
            "games_played": self.games_played,
            "epsilon": self.epsilon,
            "q_table_states": len(self.q),
        }

    # ---------- persistence ----------
    @staticmethod
    def _wrap(raw: Dict[str, Dict[str, float]]):
        return defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in raw.items()})

    def load_trained_model_from_file(self, path: str | Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.q = self._wrap(json.load(f))
            print(f"[QLearner] loaded Q‑table from '{path}'  ({len(self.q)} states)")
        except FileNotFoundError:
            print(f"[warning] Q‑table file '{path}' not found – starting fresh")

    def save_trained_model_to_file(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: dict(v) for k, v in self.q.items()}, f)
        print(f"[QLearner] saved Q‑table to '{path}'  ({len(self.q)} states)")

    def save_if_requested(self):
        if self.save_path:
            self.save_trained_model_to_file(self.save_path)
