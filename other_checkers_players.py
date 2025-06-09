"""
other_checkers_players.py
=========================

Implementation of non-learning players for checkers:
- MinimaxAlphaBetaPlayer: Perfect play using minimax with alpha-beta pruning
- RandomPlayer: Random move selection for baseline comparison
- HumanPlayer: Interface for human players with GUI support

These players serve as training opponents and comparison baselines for AI learning algorithms.
"""

import numpy as np
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from draughts import Board, Move

from checkers_player_interface import AbstractCheckersPlayer, CheckersPlayerConfiguration



class RandomCheckersPlayer(AbstractCheckersPlayer):
    """
    Random move selection player for baseline comparison and testing
    
    This player randomly selects from available legal moves.
    Useful for:
    - Baseline performance comparison
    - Initial training opponent for learning algorithms
    - Testing game system functionality
    """
    
    def __init__(self, player_configuration: CheckersPlayerConfiguration):
        super().__init__(player_configuration)
        
        # Extract random player parameters
        config_params = player_configuration.configuration_parameters
        self.random_seed = config_params.get('random_seed', None)
        self.move_delay_seconds = config_params.get('move_delay_seconds', 0.0)
        
        # Initialize random number generator
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
    
    def choose_move_from_legal_actions(self, current_board_state: np.ndarray, 
                                     available_legal_actions: List[int]) -> int:
        """
        Randomly select from available legal actions
        
        Args:
            current_board_state: Current board state (unused for random player)
            available_legal_actions: List of legal action indices
            
        Returns:
            int: Randomly selected action index
        """
        if not available_legal_actions:
            raise ValueError("No legal actions available for random selection")
        
        # Add artificial delay if configured
        if self.move_delay_seconds > 0:
            time.sleep(self.move_delay_seconds)
        
        return random.choice(available_legal_actions)
    
    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        """Update game statistics"""
        did_win = game_result_data.get('did_win', False)
        was_draw = game_result_data.get('was_draw', False)
        self.record_game_outcome(did_win, was_draw, game_result_data)


class HumanCheckersPlayer(AbstractCheckersPlayer):
    """
    Human player interface with support for GUI and console input
    
    This player allows human interaction with the checkers system.
    Supports both graphical and text-based interfaces.
    """
    
    def __init__(self, player_configuration: CheckersPlayerConfiguration):
        super().__init__(player_configuration)
        
        # Extract human player parameters
        config_params = player_configuration.configuration_parameters
        self.player_display_name = config_params.get('display_name', 'Human Player')
        self.enable_move_hints = config_params.get('enable_move_hints', False)
        self.time_limit_seconds = config_params.get('time_limit_seconds', None)
        self.use_graphical_interface = config_params.get('use_gui', True)
        
        # GUI components (will be set by GUI system)
        self.gui_interface = None
        self.move_selection_callback = None
    
    def choose_move_from_legal_actions(self, current_board_state: np.ndarray, 
                                     available_legal_actions: List[int]) -> int:
        """
        Get move selection from human player via GUI or console
        
        Args:
            current_board_state: Current board state for display
            available_legal_actions: List of legal action indices
            
        Returns:
            int: Human-selected action index
        """
        if self.use_graphical_interface and self.gui_interface:
            return self._get_move_from_gui(current_board_state, available_legal_actions)
        else:
            return self._get_move_from_console(current_board_state, available_legal_actions)
    
    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        """Update game statistics and possibly show game summary"""
        did_win = game_result_data.get('did_win', False)
        was_draw = game_result_data.get('was_draw', False)
        self.record_game_outcome(did_win, was_draw, game_result_data)
        
        # Display game result to human player
        if did_win:
            print(f"Congratulations {self.player_display_name}! You won!")
        elif was_draw:
            print(f"Game ended in a draw, {self.player_display_name}.")
        else:
            print(f"You lost this game, {self.player_display_name}. Better luck next time!")
    
    def set_gui_interface(self, gui_interface) -> None:
        """Set the GUI interface for move selection"""
        self.gui_interface = gui_interface
    
    def _get_move_from_gui(self, board_state: np.ndarray, legal_actions: List[int]) -> int:
        """Get move selection through graphical interface"""
        # This will be implemented by the GUI system
        # For now, return a placeholder
        if self.move_selection_callback:
            return self.move_selection_callback(board_state, legal_actions)
        else:
            # Fallback to console input
            return self._get_move_from_console(board_state, legal_actions)
    
    def _get_move_from_console(self, board_state: np.ndarray, legal_actions: List[int]) -> int:
        """Get move selection through console interface"""
        print("\n" + "="*50)
        print(f"{self.player_display_name}, it's your turn!")
        print("="*50)
        
        # Display current board
        self._display_board_in_console(board_state)
        
        # Show available moves if hints are enabled
        if self.enable_move_hints:
            print(f"\nAvailable moves (indices): {legal_actions}")
        
        # Get user input with validation
        while True:
            try:
                user_input = input(f"\nEnter your move index (0-{len(legal_actions)-1}): ").strip()
                selected_index = int(user_input)
                
                if 0 <= selected_index < len(legal_actions):
                    return legal_actions[selected_index]
                else:
                    print(f"Invalid index. Please enter a number between 0 and {len(legal_actions)-1}")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                # Return first legal move as default
                return legal_actions[0]
    
    def _display_board_in_console(self, board_state: np.ndarray) -> None:
        """Display the board state in console format"""
        print("\nCurrent Board Position:")
        print("   0 1 2 3 4 5 6 7")
        print("  ─────────────────")
        
        for row in range(8):
            print(f"{row}│", end="")
            for col in range(8):
                piece_value = board_state[row, col]
                if piece_value == 0:
                    print(" .", end="")
                elif piece_value == 1:
                    print(" ♦", end="")  # Your regular piece
                elif piece_value == 2:
                    print(" ♦K", end="")  # Your king (truncated for space)
                elif piece_value == -1:
                    print(" ♠", end="")  # Opponent regular piece
                elif piece_value == -2:
                    print(" ♠K", end="")  # Opponent king (truncated for space)
                
                if col < 7:
                    print(" ", end="")
            print("│")
        
        print("  ─────────────────")
        print("Legend: ♦=Your pieces, ♠=Opponent pieces, K=King")


def create_player_from_configuration(player_type: str, configuration_params: Dict[str, Any],
                                   model_file_path: Optional[str] = None) -> AbstractCheckersPlayer:
    """
    Factory function to create players of different types
    
    Args:
        player_type: Type of player to create ('qlearning', 'minimax', 'random', 'human')
        configuration_params: Configuration parameters for the player
        model_file_path: Optional path to saved model file
        
    Returns:
        AbstractCheckersPlayer: Configured player instance
    """
    player_config = CheckersPlayerConfiguration(
        player_type_name=player_type,
        configuration_parameters=configuration_params,
        model_file_path=model_file_path,
        creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    if player_type == 'minimax':
        return MinimaxAlphaBetaCheckersPlayer(player_config)
    elif player_type == 'random':
        return RandomCheckersPlayer(player_config)
    elif player_type == 'human':
        return HumanCheckersPlayer(player_config)
    elif player_type == 'qlearning':
        from qlearning_checkers_player import QLearningCheckersPlayer
        return QLearningCheckersPlayer(player_config)
    else:
        raise ValueError(f"Unknown player type: {player_type}")
