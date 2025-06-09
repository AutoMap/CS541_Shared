"""
checkers_game_environment.py
============================

Core game environment that provides a standardized interface for all AI players.
This is the main interface that neural network students will use to interact with the game.

Key Features:
- Standardized 8x8 board representation 
- Clean action space for AI agents
- Game state management and rule enforcement
- Statistics tracking for analysis
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from draughts import Board, Move
from dataclasses import dataclass


@dataclass
class CheckersGameResult:
    """Structure to store comprehensive game results for analysis"""
    winning_player_name: str  # 'player_one', 'player_two', 'draw'
    total_game_moves: int  # number of moves in the game
    final_pieces_count: Dict[str, int]  # pieces remaining for each player
    game_duration_seconds: float  # time taken for the game
    player_one_type: str  # e.g., 'qlearning', 'neural_network', 'human'
    player_two_type: str
    game_timestamp: str
    player_one_configuration: Optional[Dict] = None  # AI-specific config
    player_two_configuration: Optional[Dict] = None


class CheckersGameEnvironment:
    """
    Standardized interface for checkers game that all AI players must use.
    
    This class provides a clean, consistent API that allows different AI approaches
    (Q-learning, neural networks, minimax) to interact with the game using the same interface.
    
    State Representation:
    - 8x8 numpy array where each cell represents a board square
    - Values: -2 (opponent king), -1 (opponent piece), 0 (empty), 1 (your piece), 2 (your king)
    - Always represents the board from the current player's perspective
    """
    
    def __init__(self, enable_detailed_logging: bool = False):
        """
        Initialize the checkers game environment
        
        Args:
            enable_detailed_logging: If True, tracks detailed move history and timing
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.move_history_list = []
        self.reset_game_to_initial_state()
    
    def reset_game_to_initial_state(self) -> np.ndarray:
        """
        Reset game to initial starting position
        
        Returns:
            np.ndarray: Initial board state from current player's perspective
        """
        self.pydraughts_board = Board()
        self.is_game_finished = False
        self.winning_player_name = None
        self.total_moves_played = 0
        self.game_start_timestamp = time.time()
        self.move_history_list = []
        
        return self.get_current_board_state()
    
    def get_current_board_state(self) -> np.ndarray:
        """
        Get standardized board representation as 8x8 numpy array
        
        This is the PRIMARY INTERFACE for neural network students!
        
        Returns:
            np.ndarray: 8x8 board where:
                -2: Opponent king piece
                -1: Opponent regular piece  
                 0: Empty square
                 1: Your regular piece
                 2: Your king piece
                 
        Note: Board is always from current player's perspective
        """
        standardized_board_state = np.zeros((8, 8), dtype=int)
        
        # Get internal board representation from draughts
        internal_board_array = self.pydraughts_board.board
        current_active_player = self.pydraughts_board.whose_turn()
        
        # Convert draughts 50-square representation to our 8x8 standard
        for square_index in range(50):
            if internal_board_array[square_index] != 0:
                row_position, column_position = self._convert_square_index_to_coordinates(square_index)
                piece_type_value = internal_board_array[square_index]
                
                # Determine if this piece belongs to current player or opponent
                if piece_type_value > 0:  # White pieces (positive values)
                    if current_active_player == 1:  # White player's turn
                        # This is current player's piece
                        standardized_board_state[row_position, column_position] = 2 if piece_type_value == 3 else 1
                    else:
                        # This is opponent's piece
                        standardized_board_state[row_position, column_position] = -2 if piece_type_value == 3 else -1
                else:  # Black pieces (negative values)
                    if current_active_player == 2:  # Black player's turn
                        # This is current player's piece
                        standardized_board_state[row_position, column_position] = 2 if piece_type_value == -3 else 1
                    else:
                        # This is opponent's piece
                        standardized_board_state[row_position, column_position] = -2 if piece_type_value == -3 else -1
        
        return standardized_board_state
    
    def get_all_legal_action_indices(self) -> List[int]:
        """
        Get all legal moves as action indices
        
        Returns:
            List[int]: List of valid action indices that can be passed to execute_action_by_index()
        """
        available_legal_moves = list(self.pydraughts_board.legal_moves())
        return list(range(len(available_legal_moves)))
    
    def get_all_legal_moves_as_objects(self) -> List[Move]:
        """
        Get actual Move objects for internal processing
        
        Returns:
            List[Move]: draughts Move objects
        """
        return list(self.pydraughts_board.legal_moves())
    
    def execute_action_by_index(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute a move specified by action index
        
        Args:
            action_index: Index of the move to execute (from get_all_legal_action_indices)
            
        Returns:
            Tuple containing:
                - next_state (np.ndarray): Board state after move
                - reward (float): Immediate reward for this move
                - is_terminal (bool): Whether game has ended
                - info_dictionary (Dict): Additional information about the move
        """
        if self.is_game_finished:
            raise ValueError("Cannot execute move: game has already finished")
        
        available_legal_moves = self.get_all_legal_moves_as_objects()
        
        if action_index < 0 or action_index >= len(available_legal_moves):
            raise ValueError(f"Invalid action index {action_index}. Must be 0-{len(available_legal_moves)-1}")
        
        selected_move = available_legal_moves[action_index]
        
        # Record move if detailed logging is enabled
        if self.enable_detailed_logging:
            self.move_history_list.append({
                'move_number': self.total_moves_played + 1,
                'player': self.pydraughts_board.whose_turn(),
                'move_object': selected_move,
                'timestamp': time.time() - self.game_start_timestamp
            })
        
        # Execute the move on the board
        self.pydraughts_board.push(selected_move)
        self.total_moves_played += 1
        
        # Check if game has ended and calculate reward
        reward_value = self._calculate_move_reward()
        self.is_game_finished = self.pydraughts_board.is_over()
        
        if self.is_game_finished:
            self.winning_player_name = self._determine_winner()
        
        # Prepare information dictionary
        move_info_dictionary = {
            'move_notation': str(selected_move),
            'moves_played': self.total_moves_played,
            'captured_pieces': self._count_captured_pieces_this_move(),
            'current_player': self.pydraughts_board.whose_turn() if not self.is_game_finished else None
        }
        
        next_board_state = self.get_current_board_state()
        
        return next_board_state, reward_value, self.is_game_finished, move_info_dictionary
    
    def get_current_player_identifier(self) -> int:
        """
        Get identifier for current player (1 for white, 2 for black)
        
        Returns:
            int: Current player identifier
        """
        return self.pydraughts_board.whose_turn()
    
    def create_game_result_summary(self, player_one_name: str, player_two_name: str, 
                                 player_one_config: Optional[Dict] = None,
                                 player_two_config: Optional[Dict] = None) -> CheckersGameResult:
        """
        Create comprehensive game result summary for analysis
        
        Args:
            player_one_name: Type/name of first player
            player_two_name: Type/name of second player  
            player_one_config: Configuration used by first player
            player_two_config: Configuration used by second player
            
        Returns:
            CheckersGameResult: Complete game analysis data
        """
        game_duration = time.time() - self.game_start_timestamp
        
        # Count remaining pieces for each player
        final_piece_counts = self._count_remaining_pieces_by_player()
        
        return CheckersGameResult(
            winning_player_name=self.winning_player_name or 'draw',
            total_game_moves=self.total_moves_played,
            final_pieces_count=final_piece_counts,
            game_duration_seconds=game_duration,
            player_one_type=player_one_name,
            player_two_type=player_two_name,
            game_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            player_one_configuration=player_one_config,
            player_two_configuration=player_two_config
        )
    
    def render_board_to_console(self, show_coordinates: bool = True):
        """
        Display current board state in console
        
        Args:
            show_coordinates: Whether to show row/column coordinates
        """
        board_state = self.get_current_board_state()
        
        if show_coordinates:
            print("   ", end="")
            for col in range(8):
                print(f" {col} ", end="")
            print()
        
        for row in range(8):
            if show_coordinates:
                print(f"{row}: ", end="")
            
            for col in range(8):
                piece_value = board_state[row, col]
                if piece_value == 0:
                    print(" . ", end="")
                elif piece_value == 1:
                    print(" ♦ ", end="")  # Your regular piece
                elif piece_value == 2:
                    print(" ♦K", end="")  # Your king
                elif piece_value == -1:
                    print(" ♠ ", end="")  # Opponent regular piece
                elif piece_value == -2:
                    print(" ♠K", end="")  # Opponent king
            print()
        
        print(f"\nCurrent player: {self.get_current_player_identifier()}")
        print(f"Moves played: {self.total_moves_played}")
        print(f"Game finished: {self.is_game_finished}")
        if self.is_game_finished:
            print(f"Winner: {self.winning_player_name}")
    
    # Private helper methods
    
    def _convert_square_index_to_coordinates(self, square_index: int) -> Tuple[int, int]:
        """Convert draughts square index to 8x8 coordinates"""
        # draughts uses a specific numbering system for dark squares only
        # This conversion maps to standard 8x8 board coordinates
        row = square_index // 5
        col = (square_index % 5) * 2 + (row % 2)
        return row, col
    
    def _calculate_move_reward(self) -> float:
        """Calculate immediate reward for the move just made"""
        if self.is_game_finished:
            winner = self._determine_winner()
            if winner == 'player_one':
                return 1.0 if self.get_current_player_identifier() == 2 else -1.0
            elif winner == 'player_two':
                return 1.0 if self.get_current_player_identifier() == 1 else -1.0
            else:
                return 0.0  # Draw
        
        # Small positive reward for continuing the game
        return 0.01
    
    def _determine_winner(self) -> Optional[str]:
        """Determine the winner of the finished game"""
        if not self.is_game_finished:
            return None
        
        result = self.pydraughts_board.result()
        if result == "1-0":
            return "player_one"  # White wins
        elif result == "0-1":
            return "player_two"  # Black wins
        else:
            return "draw"
    
    def _count_captured_pieces_this_move(self) -> int:
        """Count pieces captured in the most recent move"""
        # This would require tracking piece count before/after move
        # Simplified implementation for now
        return 0
    
    def _count_remaining_pieces_by_player(self) -> Dict[str, int]:
        """Count remaining pieces for each player"""
        board_state = self.get_current_board_state()
        
        player_pieces = np.sum(board_state > 0)  # Current player's pieces
        opponent_pieces = np.sum(board_state < 0)  # Opponent's pieces
        
        if self.get_current_player_identifier() == 1:  # White's turn
            return {'player_one': player_pieces, 'player_two': opponent_pieces}
        else:  # Black's turn
            return {'player_one': opponent_pieces, 'player_two': player_pieces}
