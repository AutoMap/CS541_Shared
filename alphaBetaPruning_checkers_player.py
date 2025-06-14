# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 23:39:24 2025

@author: luisa
"""

from draughts.core import game as Game
import random


MAX_DEPTH = 5  # Set preferred depth

def evaluate_board(game: Game) -> int:
    """Basic evaluation: +100 for each player piece, -100 for each opponent piece."""
    score = 0
    for piece in game.board.pieces:
        if piece.player == 2: #game.board.player_turn:
            if piece.captured == False:
                score += 100
                if piece.king:
                    score += 50
        else:
            if piece.captured == False:
                score -= 100
                if piece.king:
                    score -= 50
    return score

def alpha_beta(game: Game, depth: int, alpha: float, beta: float, maximizing: bool):
    if depth == 0 or game.is_over():
        return evaluate_board(game), None

    best_move = None
    #legal_moves = game.get_possible_moves() # game.legal_moves() #fix to get legal moves
    legal_moves = game.legal_moves()[0]
    if maximizing:
        max_eval = float("-inf")
        for move in legal_moves:
            next_game = game.copy()
            #next_game.move(move)
            next_game.push(move)
            eval, _ = alpha_beta(next_game, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # β cut-off
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in legal_moves:
            next_game = game.copy()
            #next_game.move(move)
            next_game.push(move)
            eval, _ = alpha_beta(next_game, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # α cut-off
        return min_eval, best_move

def choose_best_move(game: Game, depth=MAX_DEPTH):
    _, move = alpha_beta(game, depth, float("-inf"), float("inf"), game.board.player_turn == 2)
    return move

def choose_random_move(game: Game, depth=MAX_DEPTH):
    legal_moves = game.legal_moves()[0]
    move = random.choice(legal_moves)
    #move = possible_moves[0]
    return move


def play_game():
    game = Game.Game()
    move_number = 0
    while not game.is_over():
        move_number += 1
        print("Move: ", move_number)
        if game.board.player_turn == 2: #draughts.Color.WHITE:
            print("Player WHITE is thinking...")
            move = choose_best_move(game)
        else:
            print("Player BLACK is thinking...")
            move = choose_best_move(game)

        print(f"Move played: {move}")
        #game.move(move)
        game.push(move)
        if move_number == 10:
            move_number += 0
        print("Score: ", evaluate_board(game))
        print(game.get_moves)

    print("Game Over!")
    print("Winner:", game.get_winner())
    return game.get_winner()
    
def play_game_against_random():
    game = Game.Game()
    move_number = 0
    while not game.is_over():
        move_number += 1
        print("Move: ", move_number)
        if game.board.player_turn == 2: #draughts.Color.WHITE:
            print("Player WHITE is thinking...")
            move = choose_best_move(game)
        else:
            print("Player BLACK is making a random move...")
            move = choose_random_move(game)

        print(f"Move played: {move}")
        game.push(move)
        if move_number == 10: #for debugging
            move_number += 0 #for debugging
        print("Score: ", evaluate_board(game))
        print(game.get_moves)

    print("Game Over!")
    print("Winner:", game.get_winner())
    return game.get_winner()
    
def alpha_beta_pruning_move(game: Game):
    depth = MAX_DEPTH
    _, move = alpha_beta(game, depth, float("-inf"), float("inf"), game.board.player_turn == 2)
    game.push(move)
    return game

