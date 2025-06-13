"""
checkers_training_configuration.py
==================================

Configuration management for training sessions and tournament setups.
Handles command-line arguments, configuration validation, and parameter storage.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class QLearningTrainingConfiguration:
    """Configuration parameters specifically for Q-Learning training sessions"""
    learning_rate_alpha: float = 0.1
    discount_factor_gamma: float = 0.9
    initial_exploration_rate_epsilon: float = 0.9
    exploration_decay_amount: float = 0.025
    exploration_decay_interval: int = 1
    minimum_exploration_rate: float = 0.01
    number_of_training_epochs: int = 100
    games_per_training_epoch: int = 100
    training_opponent_type: str = "random"  # "random", "minimax", "trainedByFile"
    input_model_file_path: Optional[str] = None
    output_model_file_path: str = "qlearning_training_results.json"
    save_progress_every_epochs: int = 10
    display_training_progress: bool = True


@dataclass
class TournamentConfiguration:
    """Configuration for running tournaments between different AI players"""
    participating_player_configurations: List[Dict[str, str]]  # [{"type": "qlearning", "path": "model.json"}]
    games_per_matchup: int = 100
    tournament_output_file_path: str = "tournament_results.json"
    enable_detailed_game_logging: bool = True
    randomize_starting_player: bool = True
    time_limit_per_game_seconds: Optional[float] = None
    tournament_name: str = "Checkers AI Tournament"


@dataclass
class PlaySessionConfiguration:
    """Configuration for human vs AI play sessions"""
    human_player_name: str = "Human Player"
    ai_opponent_type: str = "qlearning"
    ai_model_file_path: Optional[str] = None
    enable_graphical_interface: bool = True
    enable_move_suggestions: bool = False
    save_game_history: bool = True
    game_history_output_path: str = "human_vs_ai_games.json"


class CheckersConfigurationManager:
    """
    Central configuration manager for all checkers training and play modes
    
    Handles command-line argument parsing, configuration validation,
    and provides consistent interfaces for all system components
    """
    
    def __init__(self):
        self.command_line_parser = self._create_argument_parser()
    
    def parse_command_line_arguments(self, command_line_args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments and return configuration namespace
        
        Args:
            command_line_args: Optional list of command line arguments (for testing)
            
        Returns:
            argparse.Namespace: Parsed arguments
        """
        return self.command_line_parser.parse_args(command_line_args)
    
    def create_qlearning_training_config_from_args(self, parsed_args: argparse.Namespace) -> QLearningTrainingConfiguration:
        """
        Create Q-Learning training configuration from parsed command line arguments
        
        Args:
            parsed_args: Parsed command line arguments
            
        Returns:
            QLearningTrainingConfiguration: Complete training configuration
        """
        return QLearningTrainingConfiguration(
            learning_rate_alpha=parsed_args.learning_rate,
            discount_factor_gamma=parsed_args.discount_factor,
            initial_exploration_rate_epsilon=parsed_args.initial_exploration_rate,
            exploration_decay_amount=parsed_args.exploration_decay_amount,
            exploration_decay_interval=parsed_args.exploration_decay_interval,
            minimum_exploration_rate=getattr(parsed_args, 'minimum_exploration_rate', 0.01),
            number_of_training_epochs=parsed_args.number_of_epochs,
            games_per_training_epoch=parsed_args.games_per_epoch,
            training_opponent_type=parsed_args.training_opponent,
            input_model_file_path=parsed_args.input_model_path,
            output_model_file_path=parsed_args.output_model_path,
            save_progress_every_epochs=getattr(parsed_args, 'save_interval', 10),
            display_training_progress=getattr(parsed_args, 'show_progress', True)
        )
    
    def create_tournament_config_from_args(self, parsed_args: argparse.Namespace) -> TournamentConfiguration:
        """
        Create tournament configuration from command line arguments
        
        Args:
            parsed_args: Parsed command line arguments
            
        Returns:
            TournamentConfiguration: Complete tournament setup
        """
        # Parse player specifications from command line
        # Format: "qlearning:model1.json minimax:depth5 neural:student_model.json"
        participating_players = []
        
        if hasattr(parsed_args, 'tournament_players') and parsed_args.tournament_players:
            for player_specification in parsed_args.tournament_players:
                if ':' in player_specification:
                    player_type, player_config = player_specification.split(':', 1)
                    participating_players.append({
                        'player_type': player_type,
                        'configuration_path_or_params': player_config
                    })
                else:
                    # Default configuration for player type
                    participating_players.append({
                        'player_type': player_specification,
                        'configuration_path_or_params': 'default'
                    })
        
        return TournamentConfiguration(
            participating_player_configurations=participating_players,
            games_per_matchup=getattr(parsed_args, 'games_per_matchup', 100),
            tournament_output_file_path=getattr(parsed_args, 'tournament_output', 'tournament_results.json'),
            enable_detailed_game_logging=getattr(parsed_args, 'detailed_logging', True),
            randomize_starting_player=getattr(parsed_args, 'randomize_start', True),
            time_limit_per_game_seconds=getattr(parsed_args, 'time_limit', None),
            tournament_name=getattr(parsed_args, 'tournament_name', 'Checkers AI Tournament')
        )
    
    def create_play_session_config_from_args(self, parsed_args: argparse.Namespace) -> PlaySessionConfiguration:
        """
        Create play session configuration for human vs AI games
        
        Args:
            parsed_args: Parsed command line arguments
            
        Returns:
            PlaySessionConfiguration: Complete play session setup
        """
        return PlaySessionConfiguration(
            human_player_name=getattr(parsed_args, 'human_name', 'Human Player'),
            ai_opponent_type=getattr(parsed_args, 'ai_opponent', 'qlearning'),
            ai_model_file_path=getattr(parsed_args, 'ai_model_path', None),
            enable_graphical_interface=getattr(parsed_args, 'enable_gui', True),
            enable_move_suggestions=getattr(parsed_args, 'move_hints', False),
            save_game_history=getattr(parsed_args, 'save_games', True),
            game_history_output_path=getattr(parsed_args, 'game_output', 'human_vs_ai_games.json')
        )
    
    def save_configuration_to_file(self, configuration: Any, file_path: str) -> None:
        """
        Save any configuration object to JSON file with metadata
        
        Args:
            configuration: Configuration object to save
            file_path: Where to save the configuration
        """
        save_data = {
            'configuration_metadata': {
                'creation_timestamp': datetime.now().isoformat(),
                'configuration_type': type(configuration).__name__,
                'saved_by': 'CheckersConfigurationManager'
            },
            'configuration_parameters': asdict(configuration)
        }
        
        with open(file_path, 'w') as file_handle:
            json.dump(save_data, file_handle, indent=2, default=str)
    
    def load_configuration_from_file(self, file_path: str, expected_config_type: type) -> Any:
        """
        Load configuration from JSON file with validation
        
        Args:
            file_path: Path to configuration file
            expected_config_type: Expected configuration class type
            
        Returns:
            Configuration object of the expected type
        """
        with open(file_path, 'r') as file_handle:
            loaded_data = json.load(file_handle)
        
        config_params = loaded_data['configuration_parameters']
        return expected_config_type(**config_params)
    
    def validate_configuration_parameters(self, configuration: Any) -> List[str]:
        """
        Validate configuration parameters and return list of any issues
        
        Args:
            configuration: Configuration object to validate
            
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        validation_errors = []
        
        if isinstance(configuration, QLearningTrainingConfiguration):
            validation_errors.extend(self._validate_qlearning_config(configuration))
        elif isinstance(configuration, TournamentConfiguration):
            validation_errors.extend(self._validate_tournament_config(configuration))
        elif isinstance(configuration, PlaySessionConfiguration):
            validation_errors.extend(self._validate_play_session_config(configuration))
        
        return validation_errors
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive command line argument parser"""
        main_parser = argparse.ArgumentParser(
            description="Checkers AI Training and Competition System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train Q-Learning agent
  python checkers_main.py train --learning-rate 0.1 --number-of-epochs 50
  
  # Run tournament
  python checkers_main.py tournament --players qlearning:model1.json minimax:depth5
  
  # Play against AI
  python checkers_main.py play --ai-opponent qlearning --ai-model model.json --gui
            """
        )
        
        # Add subparsers for different modes
        subparsers = main_parser.add_subparsers(dest='operation_mode', help='Operation mode')
        
        # Training mode arguments
        training_parser = subparsers.add_parser('train', help='Train AI players')
        self._add_training_arguments(training_parser)
        
        # Tournament mode arguments
        tournament_parser = subparsers.add_parser('tournament', help='Run AI tournament')
        self._add_tournament_arguments(tournament_parser)
        
        # Play mode arguments
        play_parser = subparsers.add_parser('play', help='Play against AI')
        self._add_play_arguments(play_parser)
        
        return main_parser
    
    def _add_training_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add training-specific command line arguments"""
        parser.add_argument("--player-type", type=str, default="qlearning",
                           choices=["qlearning"],
                           help="Type of AI player to train")
        
        parser.add_argument("--learning-rate", type=float, default=0.1,
                           help="Learning rate (alpha) for Q-learning (0-1)")
        
        parser.add_argument("--discount-factor", type=float, default=0.9,
                           help="Discount factor (gamma) for future rewards (0-1)")
        
        parser.add_argument("--initial-exploration-rate", type=float, default=0.9,
                           help="Initial exploration rate (epsilon) (0-1)")
        
        parser.add_argument("--exploration-decay-amount", type=float, default=0.025,
                           help="Amount to decrease exploration rate each decay")
        
        parser.add_argument("--exploration-decay-interval", type=int, default=1,
                           help="Epochs between exploration rate decreases")
        
        parser.add_argument("--minimum-exploration-rate", type=float, default=0.01,
                           help="Minimum exploration rate (never goes below this)")
        
        parser.add_argument("--number-of-epochs", type=int, default=100,
                           help="Total number of training epochs")
        
        parser.add_argument("--games-per-epoch", type=int, default=100,
                           help="Training games per epoch")
        
        parser.add_argument("--training-opponent", type=str,
                           choices=["minimax", "random", "trainedByFile"], default="random",
                           help="Opponent type for training")
        
        parser.add_argument("--input-model-path", type=str, default=None,
                           help="Path to existing model to continue training")
        
        parser.add_argument("--output-model-path", type=str, default="training_results.json",
                           help="Path to save the trained model")
        
        parser.add_argument("--save-interval", type=int, default=10,
                           help="Save progress every N epochs")
        
        parser.add_argument("--show-progress", action="store_true", default=True,
                           help="Display training progress")
    
    def _add_tournament_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add tournament-specific command line arguments"""
        parser.add_argument("--tournament-players", nargs='+', required=True,
                           help="Player specifications: type:config (e.g., qlearning:model.json)")
        
        parser.add_argument("--games-per-matchup", type=int, default=100,
                           help="Number of games per player matchup")
        
        parser.add_argument("--tournament-output", type=str, default="tournament_results.json",
                           help="Output file for tournament results")
        
        parser.add_argument("--detailed-logging", action="store_true", default=True,
                           help="Enable detailed game logging")
        
        parser.add_argument("--randomize-start", action="store_true", default=True,
                           help="Randomize starting player for each game")
        
        parser.add_argument("--time-limit", type=float, default=None,
                           help="Time limit per game in seconds")
        
        parser.add_argument("--tournament-name", type=str, default="Checkers AI Tournament",
                           help="Name for this tournament")
    
    def _add_play_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add play session arguments"""
        parser.add_argument("--human-name", type=str, default="Human Player",
                           help="Name for the human player")
        
        parser.add_argument("--ai-opponent", type=str, default="qlearning",
                           choices=["qlearning", "minimax", "neural"],
                           help="Type of AI opponent")
        
        parser.add_argument("--ai-model-path", type=str, default=None,
                           help="Path to AI model file")
        
        parser.add_argument("--enable-gui", action="store_true", default=True,
                           help="Enable graphical user interface")
        
        parser.add_argument("--move-hints", action="store_true", default=False,
                           help="Enable move suggestions for human player")
        
        parser.add_argument("--save-games", action="store_true", default=True,
                           help="Save game history to file")
        
        parser.add_argument("--game-output", type=str, default="human_vs_ai_games.json",
                           help="Output file for game history")
    
    def _validate_qlearning_config(self, config: QLearningTrainingConfiguration) -> List[str]:
        """Validate Q-Learning specific configuration parameters"""
        errors = []
        
        if not (0.0 <= config.learning_rate_alpha <= 1.0):
            errors.append("Learning rate must be between 0.0 and 1.0")
        
        if not (0.0 <= config.discount_factor_gamma <= 1.0):
            errors.append("Discount factor must be between 0.0 and 1.0")
        
        if not (0.0 <= config.initial_exploration_rate_epsilon <= 1.0):
            errors.append("Initial exploration rate must be between 0.0 and 1.0")
        
        if not (0.0 <= config.minimum_exploration_rate <= config.initial_exploration_rate_epsilon):
            errors.append("Minimum exploration rate must be <= initial exploration rate")
        
        if config.exploration_decay_amount < 0:
            errors.append("Exploration decay amount must be non-negative")
        
        if config.exploration_decay_interval < 1:
            errors.append("Exploration decay interval must be at least 1")
        
        if config.number_of_training_epochs < 1:
            errors.append("Number of training epochs must be at least 1")
        
        if config.games_per_training_epoch < 1:
            errors.append("Games per epoch must be at least 1")
        
        if config.training_opponent_type not in ["random", "minimax", "trainedByFile"]:
            errors.append("Training opponent must be 'random', 'minimax', or 'trainedByFile'")
        
        if config.save_progress_every_epochs < 1:
            errors.append("Save interval must be at least 1 epoch")
        
        return errors
    
    def _validate_tournament_config(self, config: TournamentConfiguration) -> List[str]:
        """Validate tournament configuration parameters"""
        errors = []
        
        if len(config.participating_player_configurations) < 2:
            errors.append("Tournament requires at least 2 players")
        
        if config.games_per_matchup < 1:
            errors.append("Games per matchup must be at least 1")
        
        if config.time_limit_per_game_seconds is not None and config.time_limit_per_game_seconds <= 0:
            errors.append("Time limit must be positive if specified")
        
        # Validate player specifications
        for i, player_config in enumerate(config.participating_player_configurations):
            if 'player_type' not in player_config:
                errors.append(f"Player {i+1} missing player_type specification")
            elif player_config['player_type'] not in ['qlearning', 'minimax', 'neural', 'human', 'random']:
                errors.append(f"Player {i+1} has invalid player_type: {player_config['player_type']}")
        
        return errors
    
    def _validate_play_session_config(self, config: PlaySessionConfiguration) -> List[str]:
        """Validate play session configuration parameters"""
        errors = []
        
        if not config.human_player_name.strip():
            errors.append("Human player name cannot be empty")
        
        if config.ai_opponent_type not in ['qlearning', 'minimax', 'neural', 'random']:
            errors.append("AI opponent type must be 'qlearning', 'minimax', 'neural', or 'random'")
        
        return errors