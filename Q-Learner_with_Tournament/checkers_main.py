#!/usr/bin/env python3
"""
Top‑level training / tournament runner for English draughts.

Only the single‑game loop (_play_one_game) has been expanded so that it
carries **turn information** in parallel with each board state.
"""

from __future__ import annotations

import argparse
import textwrap
import time
from typing import Any, Dict, List

from checkers_game_environment import CheckersGameEnvironment
from checkers_player_interface import CheckersPlayerConfiguration
from other_checkers_players import create_player_from_configuration
from qlearning_checkers_player import QLearningCheckersPlayer


# ───────────────────────── helpers (unchanged) ─────────────────────────
def _coerce(s: str) -> Any:
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            continue
    return s


def _parse_opts(block: str) -> Dict[str, Any]:
    return {} if not block else {
        k.strip(): _coerce(v.strip())
        for k, v in (pair.split("=", 1) for pair in block.split(",") if "=" in pair)
    }


def _build_player(spec: str):
    if "[" in spec:
        base, opts = spec.split("[", 1)
        opts = _parse_opts(opts.rstrip("]"))
    else:
        base, opts = spec, {}
    base = base.strip().lower()

    if base in {"random", "minimax"}:
        return create_player_from_configuration(base, opts)

    if base == "qlearning":
        load_path = opts.pop("in", None)
        save_path = opts.pop("out", None)
        cfg = CheckersPlayerConfiguration(
            player_type_name="qlearning",
            configuration_parameters={**opts, "save_path": save_path},
            model_file_path=load_path,
        )
        return QLearningCheckersPlayer(cfg)

    raise ValueError(f"Unknown player type '{base}'")


# ───────────────────── single game ─────────────────────
def _play_one_game(p1, p2, env: CheckersGameEnvironment, train: bool):
    state, turn = env.reset_game_to_initial_state()

    players = [p1, p2]
    idx = 0                       # 0 = player_one, 1 = player_two
    reward = 0.0

    last_state = [state, state]
    last_turn  = [turn,  turn]
    last_action = [None, None]

    while not env.is_game_finished:
        legal = env.get_all_legal_action_indices()

        # Hand current turn to the chooser without changing the public API
        setattr(players[idx], "_current_turn", turn)

        action = players[idx].choose_move_from_legal_actions(state, legal)

        last_state[idx]  = state
        last_turn[idx]   = turn
        last_action[idx] = action

        # Update the *previous* mover with reward 0 for a non‑terminal ply
        prev = 1 - idx
        if (
            train
            and last_action[prev] is not None
            and isinstance(players[prev], QLearningCheckersPlayer)
        ):
            players[prev].learn_one_transition(
                last_state[prev],
                last_turn[prev],
                last_action[prev],
                reward,
                state,            # next_board
                turn,             # next_turn
                legal,
            )

        # Make the move
        (state, turn), reward, done, _ = env.execute_action_by_index(action)

        # Immediate terminal update for the side that just moved
        if train and done and isinstance(players[idx], QLearningCheckersPlayer):
            players[idx].learn_one_transition(
                last_state[idx],
                last_turn[idx],
                action,
                reward,
                state,
                turn,
                [],
            )

        idx = 1 - idx

    return env.winning_player_name, env.total_moves_played


# ─────────────────────────── main (unchanged below) ──────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=textwrap.dedent(__doc__),
        add_help=False,
    )

    p.add_argument("player1")
    p.add_argument("player2")
    p.add_argument("-n", "--total-games", type=int)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--games-per-epoch", type=int, default=1)
    p.add_argument("--train", action="store_true")
    p.add_argument("-h", "--help", action="help")
    p.add_argument("-help", action="help", help=argparse.SUPPRESS)

    a = p.parse_args()
    total_games = a.total_games or a.epochs * a.games_per_epoch
    if total_games <= 0:
        raise SystemExit("total games computed as 0 — nothing to do")

    env = CheckersGameEnvironment()
    p1 = _build_player(a.player1)
    p2 = _build_player(a.player2)

    p1_wins = p2_wins = draws = 0
    start = time.time()

    g = 0
    for epoch in range(1, a.epochs + 1):
        ep_p1 = ep_p2 = ep_draws = ep_moves = 0
        for _ in range(a.games_per_epoch):
            if g >= total_games:
                break
            g += 1
            winner, moves = _play_one_game(p1, p2, env, train=a.train)
            ep_moves += moves

            for pl in (p1, p2):
                if isinstance(pl, QLearningCheckersPlayer):
                    pl.update_after_game_completion({})

            if winner == "player_one":
                p1_wins += 1; ep_p1 += 1
            elif winner == "player_two":
                p2_wins += 1; ep_p2 += 1
            else:
                draws += 1; ep_draws += 1

        games_this_epoch = ep_p1 + ep_p2 + ep_draws
        avg_moves = ep_moves / games_this_epoch if games_this_epoch else 0.0
        epsilons = [
            f"{pl.epsilon:.2f}"
            for pl in (p1, p2)
            if isinstance(pl, QLearningCheckersPlayer)
        ]
        print(
            f"Epoch {epoch}/{a.epochs} — P1:{ep_p1}  P2:{ep_p2}  draw:{ep_draws}  "
            f"avg_moves:{avg_moves:.1f}  ε:{'/'.join(epsilons) or 'N/A'}"
        )

    elapsed = time.time() - start
    print("\nFinal summary")
    print(f"  {p1.get_player_type_name()} wins: {p1_wins}")
    print(f"  {p2.get_player_type_name()} wins: {p2_wins}")
    print(f"  draws: {draws}")
    print(f"  elapsed: {elapsed:.1f}s")

    for pl in (p1, p2):
        if isinstance(pl, QLearningCheckersPlayer):
            pl.save_if_requested()


if __name__ == "__main__":
    main()
