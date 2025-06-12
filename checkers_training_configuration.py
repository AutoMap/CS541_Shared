#!/usr/bin/env python3
"""
checkers_training_configuration.py
==================================

Configuration management for the Checkers AI Training and Competition System.

This module provides:
- Command-line argument parsing
- Configuration validation
- Configuration object creation for different system modes
"""

import argparse
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime


class QLearningTrainingConfiguration:
    """Configuration parameters for Q-Learning training sessions"""
    
    def __init__(self):
        # Core learning parameters
        self.player_type_name = 'neural_network'  # 'qlearning' or 'neural_network'
        self.learning_rate_alpha = 0.1
        self.discount_factor_gamma = 0.9
        self.initial_exploration_rate_epsilon = 1.0
        self.exploration_decay_amount = 0.01
        self.exploration_decay_interval = 100  # games
        self.minimum_exploration_rate = 0.05
        
        # Training session parameters
        self.number_of_training_epochs = 50
        self.games_per_training_epoch = 100
        self.training_opponent_type = 'random'  # random, minimax, trainedByFile
        self.display_training_progress = True
        self.save_progress_every_epochs = 10
        
        # File paths
        self.input_model_file_path = None  # Optional path to continue training from
        self.output_model_file_path = f"trained_model_{int(datetime.now().timestamp())}.json"


class TournamentConfiguration:
    """Configuration parameters for tournament sessions"""
    
    def __init__(self):
        self.tournament_name = f"Tournament_{int(datetime.now().timestamp())}"
        self.participating_player_configurations = []  # List of players
        self.games_per_matchup = 10
        self.randomize_starting_player = True
        self.time_limit_per_game_seconds = 30  # Optional time limit
        self.tournament_output_file_path = f"tournament_results_{int(datetime.now().timestamp())}.json"


class PlaySessionConfiguration:
    """Configuration parameters for human play sessions"""
    
    def __init__(self):
        self.enable_gui = True
        self.ai_opponent_type = 'qlearning'
        self.ai_model_path = None
        self.ai_configuration_params = {}
        self.human_player_goes_first = True
        self.time_limit_per_move_seconds = 0  # 0 means no limit


class CheckersConfigurationManager:
    """
    Manages configuration for all system operation modes
    
    Responsibilities:
    - Command-line argument parsing
    - Configuration validation
    - Configuration object creation
    """
    
    def __init__(self):
        self.command_line_parser = self._create_argument_parser()
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create command-line argument parser with all supported options"""
        parser = argparse.ArgumentParser(
            description="Checkers AI Training and Competition System",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Operation mode subparsers
        subparsers = parser.add_subparsers(dest='operation_mode', help='Operation mode')
        
        # Training mode parser
        train_parser = subparsers.add_parser('train', help='Train a Q-Learning agent')
        train_parser.add_argument('--learning-rate', type=float, default=0.1, 
                                help='Learning rate alpha [0.0-1.0]')
        train_parser.add_argument('--discount-factor', type=float, default=0.9, 
                                help='Discount factor gamma [0.0-1.0]')
        train_parser.add_argument('--exploration-rate', type=float, default=1.0, 
                                help='Initial exploration rate epsilon [0.0-1.0]')
        train_parser.add_argument('--number-of-epochs', type=int, default=50, 
                                help='Number of training epochs')
        train_parser.add_argument('--games-per-epoch', type=int, default=100, 
                                help='Number of games per training epoch')
        train_parser.add_argument('--training-opponent', type=str, default='random', 
                                choices=['random', 'minimax', 'trainedByFile'],
                                help='Type of opponent to train against')
        train_parser.add_argument('--input-model-path', type=str, default=None, 
                                help='Path to existing model to continue training')
        train_parser.add_argument('--output-model-path', type=str, default=None, 
                                help='Path to save trained model')
        train_parser.add_argument('--player-type', type=str, default='qlearning',
                                choices=['qlearning', 'neural_network'],
                                help='Type of learning player')
        
        # Tournament mode parser
        tournament_parser = subparsers.add_parser('tournament', 
                                               help='Run tournament between multiple AI types')
        tournament_parser.add_argument('--tournament-name', type=str, default=None, 
                                    help='Name of the tournament')
        tournament_parser.add_argument('--tournament-players', type=str, nargs='+', required=True, 
                                    help='List of players in format type:config_or_path')
        tournament_parser.add_argument('--games-per-matchup', type=int, default=10, 
                                    help='Number of games per player matchup')
        tournament_parser.add_argument('--output-file', type=str, default=None, 
                                    help='Path to save tournament results')
        
        # Play mode parser
        play_parser = subparsers.add_parser('play', help='Play against an AI opponent')
        play_parser.add_argument('--ai-opponent', type=str, default='qlearning', 
                              choices=['qlearning', 'minimax', 'random'],
                              help='Type of AI opponent')
        play_parser.add_argument('--ai-model-path', type=str, default=None, 
                              help='Path to AI model (for trained models)')
        play_parser.add_argument('--enable-gui', action='store_true', 
                              help='Enable GUI for gameplay')
        play_parser.add_argument('--human-first', action='store_true', 
                              help='Human player goes first')
        
        return parser
    
    def parse_command_line_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments"""
        return self.command_line_parser.parse_args()
    
    def create_qlearning_training_config_from_args(self, args) -> QLearningTrainingConfiguration:
        """Create training configuration from command-line arguments"""
        config = QLearningTrainingConfiguration()
        
        if hasattr(args, 'player_type'):
            config.player_type_name = args.player_type
            
        if hasattr(args, 'learning_rate'):
            config.learning_rate_alpha = args.learning_rate
            
        if hasattr(args, 'discount_factor'):
            config.discount_factor_gamma = args.discount_factor
            
        if hasattr(args, 'exploration_rate'):
            config.initial_exploration_rate_epsilon = args.exploration_rate
            
        if hasattr(args, 'number_of_epochs'):
            config.number_of_training_epochs = args.number_of_epochs
            
        if hasattr(args, 'games_per_epoch'):
            config.games_per_training_epoch = args.games_per_epoch
            
        if hasattr(args, 'training_opponent'):
            config.training_opponent_type = args.training_opponent
            
        if hasattr(args, 'input_model_path') and args.input_model_path:
            config.input_model_file_path = args.input_model_path
            
        if hasattr(args, 'output_model_path') and args.output_model_path:
            config.output_model_file_path = args.output_model_path
            
        return config
    
    def create_tournament_config_from_args(self, args) -> TournamentConfiguration:
        """Create tournament configuration from command-line arguments"""
        config = TournamentConfiguration()
        
        if hasattr(args, 'tournament_name') and args.tournament_name:
            config.tournament_name = args.tournament_name
            
        if hasattr(args, 'games_per_matchup'):
            config.games_per_matchup = args.games_per_matchup
            
        if hasattr(args, 'output_file') and args.output_file:
            config.tournament_output_file_path = args.output_file
            
        # Parse player specifications
        if hasattr(args, 'tournament_players'):
            for player_spec in args.tournament_players:
                parts = player_spec.split(':')
                player_type = parts[0]
                config_path_or_params = parts[1] if len(parts) > 1 else 'default'
                
                config.participating_player_configurations.append({
                    'player_type': player_type,
                    'configuration_path_or_params': config_path_or_params
                })
            
        return config
    
    def create_play_session_config_from_args(self, args) -> PlaySessionConfiguration:
        """Create play session configuration from command-line arguments"""
        config = PlaySessionConfiguration()
        
        if hasattr(args, 'enable_gui'):
            config.enable_gui = args.enable_gui
            
        if hasattr(args, 'ai_opponent'):
            config.ai_opponent_type = args.ai_opponent
            
        if hasattr(args, 'ai_model_path') and args.ai_model_path:
            config.ai_model_path = args.ai_model_path
            
        if hasattr(args, 'human_first'):
            config.human_player_goes_first = args.human_first
            
        return config
    
    def validate_configuration_parameters(self, config) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if isinstance(config, QLearningTrainingConfiguration):
            # Validate Q-Learning training configuration
            if not 0.0 <= config.learning_rate_alpha <= 1.0:
                errors.append("Learning rate must be between 0.0 and 1.0")
                
            if not 0.0 <= config.discount_factor_gamma <= 1.0:
                errors.append("Discount factor must be between 0.0 and 1.0")
                
            if not 0.0 <= config.initial_exploration_rate_epsilon <= 1.0:
                errors.append("Exploration rate must be between 0.0 and 1.0")
                
            if config.number_of_training_epochs <= 0:
                errors.append("Number of training epochs must be positive")
                
            if config.games_per_training_epoch <= 0:
                errors.append("Games per epoch must be positive")
                
            if config.input_model_file_path and not os.path.exists(config.input_model_file_path):
                errors.append(f"Input model file not found: {config.input_model_file_path}")
                
        elif isinstance(config, TournamentConfiguration):
            # Validate tournament configuration
            if not config.participating_player_configurations:
                errors.append("Tournament requires at least two participating players")
            elif len(config.participating_player_configurations) < 2:
                errors.append("Tournament requires at least two participating players")
                
            if config.games_per_matchup <= 0:
                errors.append("Games per matchup must be positive")
                
        elif isinstance(config, PlaySessionConfiguration):
            # Validate play session configuration
            if config.ai_model_path and not os.path.exists(config.ai_model_path):
                errors.append(f"AI model file not found: {config.ai_model_path}")
        
        return errors