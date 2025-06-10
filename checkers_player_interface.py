"""
checkers_player_interface.py
============================

Abstract base class and interfaces for all checkers players.
This defines the standard interface that all AI implementations must follow.

This is the KEY INTERFACE that neural network students must implement!
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CheckersPlayerConfiguration:
    """
    Configuration data structure for player settings
    This allows easy serialization and comparison of different player setups
    """
    player_type_name: str  # e.g., 'qlearning', 'neural_network', 'minimax'
    configuration_parameters: Dict[str, Any]  # Type-specific parameters
    model_file_path: Optional[str] = None  # Path to saved model/weights
    creation_timestamp: Optional[str] = None
    training_information: Optional[Dict] = None  # Training history, epochs, etc.


class AbstractCheckersPlayer(ABC):
    """
    Abstract base class that ALL checkers players must inherit from.
    
    This provides a standardized interface that allows different AI approaches
    to compete against each other seamlessly.
    
    NEURAL NETWORK STUDENTS: You must implement this interface!
    """
    
    def __init__(self, player_configuration: CheckersPlayerConfiguration):
        """
        Initialize the player with configuration
        
        Args:
            player_configuration: Configuration object containing player settings
        """
        self.player_configuration = player_configuration
        self.total_games_played = 0
        self.total_wins_achieved = 0
        self.total_losses_suffered = 0
        self.total_draws_reached = 0
        self.game_statistics_history = []
    
    @abstractmethod
    def choose_move_from_legal_actions(self, current_board_state: np.ndarray, 
                                     available_legal_actions: List[int]) -> int:
        """
        Choose the best move given current board state and legal actions.
        
        THIS IS THE MAIN METHOD that neural network students must implement!
        
        Args:
            current_board_state: 8x8 numpy array representing current board position
                                Values: -2 (opp king), -1 (opp piece), 0 (empty), 
                                       1 (your piece), 2 (your king)
            available_legal_actions: List of valid action indices to choose from
            
        Returns:
            int: Index of chosen action (must be in available_legal_actions list)
        """
        pass
    
    @abstractmethod
    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        """
        Update player after game completion (for learning algorithms)
        
        Args:
            game_result_data: Dictionary containing game outcome and statistics
        """
        pass
    
    def get_player_type_name(self) -> str:
        """Get the type name of this player"""
        return self.player_configuration.player_type_name
    
    def get_player_statistics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this player's performance
        
        Returns:
            Dict containing win rate, games played, and other metrics
        """
        if self.total_games_played == 0:
            win_rate_percentage = 0.0
        else:
            win_rate_percentage = (self.total_wins_achieved / self.total_games_played) * 100
        
        return {
            'player_type': self.get_player_type_name(),
            'total_games_played': self.total_games_played,
            'wins_achieved': self.total_wins_achieved,
            'losses_suffered': self.total_losses_suffered,
            'draws_reached': self.total_draws_reached,
            'win_rate_percentage': win_rate_percentage,
            'configuration_used': self.player_configuration.configuration_parameters
        }
    
    def record_game_outcome(self, did_win: bool, was_draw: bool, 
                           additional_game_data: Optional[Dict] = None) -> None:
        """
        Record the outcome of a completed game
        
        Args:
            did_win: True if this player won the game
            was_draw: True if the game was a draw
            additional_game_data: Optional additional statistics about the game
        """
        self.total_games_played += 1
        
        if was_draw:
            self.total_draws_reached += 1
        elif did_win:
            self.total_wins_achieved += 1
        else:
            self.total_losses_suffered += 1
        
        # Store game data for detailed analysis
        game_record = {
            'game_number': self.total_games_played,
            'outcome': 'win' if did_win else ('draw' if was_draw else 'loss'),
            'additional_data': additional_game_data or {}
        }
        self.game_statistics_history.append(game_record)
    
    def save_player_configuration_to_file(self, file_path: str) -> None:
        """
        Save player configuration and statistics to JSON file
        
        Args:
            file_path: Path where to save the configuration file
        """
        import json
        from datetime import datetime
        
        save_data = {
            'configuration': {
                'player_type_name': self.player_configuration.player_type_name,
                'configuration_parameters': self.player_configuration.configuration_parameters,
                'model_file_path': self.player_configuration.model_file_path,
                'creation_timestamp': self.player_configuration.creation_timestamp,
                'training_information': self.player_configuration.training_information
            },
            'statistics': self.get_player_statistics_summary(),
            'game_history': self.game_statistics_history,
            'file_saved_timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as file_handle:
            json.dump(save_data, file_handle, indent=2, default=str)
    
    @classmethod
    def load_player_configuration_from_file(cls, file_path: str) -> 'CheckersPlayerConfiguration':
        """
        Load player configuration from JSON file
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            CheckersPlayerConfiguration: Loaded configuration object
        """
        import json
        
        with open(file_path, 'r') as file_handle:
            loaded_data = json.load(file_handle)
        
        config_data = loaded_data['configuration']
        return CheckersPlayerConfiguration(
            player_type_name=config_data['player_type_name'],
            configuration_parameters=config_data['configuration_parameters'],
            model_file_path=config_data.get('model_file_path'),
            creation_timestamp=config_data.get('creation_timestamp'),
            training_information=config_data.get('training_information')
        )


class AbstractTrainableCheckersPlayer(AbstractCheckersPlayer):
    """
    Extended interface for players that can be trained through self-play
    
    Q-Learning and Neural Network players should inherit from this class
    """
    
    @abstractmethod
    def update_from_game_experience(self, game_state_sequence: List[np.ndarray],
                                  action_sequence: List[int],
                                  reward_sequence: List[float]) -> None:
        """
        Update the player's knowledge based on a complete game experience
        
        Args:
            game_state_sequence: List of board states throughout the game
            action_sequence: List of actions taken throughout the game
            reward_sequence: List of rewards received throughout the game
        """
        pass
    
    @abstractmethod
    def save_trained_model_to_file(self, file_path: str) -> None:
        """
        Save the trained model/weights to a file
        
        Args:
            file_path: Where to save the trained model
        """
        pass
    
    @abstractmethod
    def load_trained_model_from_file(self, file_path: str) -> None:
        """
        Load a previously trained model/weights from file
        
        Args:
            file_path: Path to the trained model file
        """
        pass
    
    @abstractmethod
    def get_training_progress_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about training progress
        
        Returns:
            Dict: Training metrics like learning rate, epochs completed, etc.
        """
        pass