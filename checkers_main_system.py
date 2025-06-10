#!/usr/bin/env python3
"""
checkers_main_system.py
=======================

Main entry point for the Checkers AI Training and Competition System.

This system provides a comprehensive platform for:
- Training Q-Learning agents against various opponents
- Running tournaments between different AI approaches
- Human vs AI gameplay with GUI support
- Neural network integration interface for students
- Comprehensive performance analysis and reporting

Usage Examples:
  # Train Q-Learning agent for 50 epochs
  python checkers_main_system.py train --learning-rate 0.1 --number-of-epochs 50 --training-opponent random
  
  # Continue training from existing model
  python checkers_main_system.py train --input-model-path trained_model.json --number-of-epochs 25
  
  # Run tournament between multiple AI types
  python checkers_main_system.py tournament --tournament-players qlearning:model1.json minimax:depth5 random:default
  
  # Play against trained AI with GUI
  python checkers_main_system.py play --ai-opponent qlearning --ai-model-path best_model.json --enable-gui
"""

import sys
import os
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import all system components
from checkers_game_environment import CheckersGameEnvironment, CheckersGameResult
from checkers_player_interface import AbstractCheckersPlayer, AbstractTrainableCheckersPlayer
from qlearning_checkers_player import QLearningCheckersPlayer
from other_checkers_players import (create_player_from_configuration)
from checkers_training_configuration import (CheckersConfigurationManager, QLearningTrainingConfiguration,
                                           TournamentConfiguration)
from neural_qlearning_agent import NeuralNetworkCheckersPlayer


class CheckersTrainingSystem:
    """
    Core training system for Q-Learning and other trainable players
    
    Manages the complete training pipeline including:
    - Opponent creation and management
    - Training progress tracking and display
    - Model saving and loading
    - Performance statistics collection
    """
    
    def __init__(self, training_configuration: QLearningTrainingConfiguration):
        self.training_config = training_configuration
        self.game_environment = CheckersGameEnvironment(enable_detailed_logging=True)
        
        # Training progress tracking
        self.training_start_timestamp = None
        self.epoch_statistics_history = []
        self.overall_training_statistics = {
            'total_games_played': 0,
            'total_training_time': 0.0,
            'best_win_rate_achieved': 0.0,
            'training_configuration_used': training_configuration
        }
    
    def execute_qlearning_training_session(self) -> Dict[str, Any]:
        """
        Execute complete Q-Learning training session
        
        Returns:
            Dict: Comprehensive training results and statistics
        """
        print("="*80)
        print("STARTING Q-LEARNING TRAINING SESSION")
        print("="*80)
        
        self.training_start_timestamp = time.time()
        
        # Create Q-Learning player
        if self.training_config.player_type_name == 'qlearning':
            qlearning_player = self._create_qlearning_player_from_config()
        elif self.training_config.player_type_name == 'neural_network':
            qlearning_player = self._create_neural_network_player_from_config()
        else:
            raise ValueError("Unsupported training player type.")
        
        # Create training opponent
        training_opponent = self._create_training_opponent()
        
        print(f"Training Configuration:")
        print(f"  - Learning Rate: {self.training_config.learning_rate_alpha}")
        print(f"  - Discount Factor: {self.training_config.discount_factor_gamma}")
        print(f"  - Initial Exploration Rate: {self.training_config.initial_exploration_rate_epsilon}")
        print(f"  - Training Epochs: {self.training_config.number_of_training_epochs}")
        print(f"  - Games per Epoch: {self.training_config.games_per_training_epoch}")
        print(f"  - Training Opponent: {self.training_config.training_opponent_type}")
        print(f"  - Output File: {self.training_config.output_model_file_path}")
        print()
        
        # Execute training epochs
        for current_epoch in range(self.training_config.number_of_training_epochs):
            epoch_start_time = time.time()
            
            # Run games for this epoch
            epoch_game_results = self._execute_training_epoch(qlearning_player, training_opponent)
            
            # Complete epoch and get statistics
            epoch_stats = qlearning_player.complete_training_epoch()
            epoch_stats['epoch_duration_seconds'] = time.time() - epoch_start_time
            epoch_stats['games_results'] = epoch_game_results
            
            self.epoch_statistics_history.append(epoch_stats)
            
            # Display progress if enabled
            if self.training_config.display_training_progress:
                self._display_epoch_progress(current_epoch + 1, epoch_stats, epoch_game_results)
            
            # Save progress periodically
            if ((current_epoch + 1) % self.training_config.save_progress_every_epochs == 0 or 
                current_epoch == self.training_config.number_of_training_epochs - 1):
                
                progress_save_path = f"progress_{current_epoch + 1}_{self.training_config.output_model_file_path}"
                qlearning_player.save_trained_model_to_file(progress_save_path)
                print(f"Progress saved to: {progress_save_path}")
        
        # Training completed - save final model and generate report
        training_end_time = time.time()
        total_training_duration = training_end_time - self.training_start_timestamp
        
        # Save final trained model
        qlearning_player.save_trained_model_to_file(self.training_config.output_model_file_path)
        
        # Generate comprehensive training report
        final_training_results = self._generate_final_training_report(qlearning_player, total_training_duration)
        
        print("\n" + "="*80)
        print("TRAINING SESSION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Final model saved to: {self.training_config.output_model_file_path}")
        print(f"Total training time: {total_training_duration:.2f} seconds")
        print(f"Total games played: {qlearning_player.total_games_played}")
        print(f"Final win rate: {qlearning_player.total_wins_achieved / max(1, qlearning_player.total_games_played) * 100:.1f}%")
        
        return final_training_results
    
    def _create_qlearning_player_from_config(self) -> QLearningCheckersPlayer:
        """Create Q-Learning player from training configuration"""
        from checkers_player_interface import CheckersPlayerConfiguration
        
        player_config = CheckersPlayerConfiguration(
            player_type_name='qlearning',
            configuration_parameters={
                'learning_rate_alpha': self.training_config.learning_rate_alpha,
                'discount_factor_gamma': self.training_config.discount_factor_gamma,
                'exploration_rate_epsilon': self.training_config.initial_exploration_rate_epsilon,
                'exploration_rate_min': self.training_config.minimum_exploration_rate
            },
            model_file_path=self.training_config.input_model_file_path,
            creation_timestamp=datetime.now().isoformat()
        )
        
        return QLearningCheckersPlayer(player_config)
    
    def _create_neural_network_player_from_config(self) -> AbstractTrainableCheckersPlayer:
        from checkers_player_interface import CheckersPlayerConfiguration

        player_config = CheckersPlayerConfiguration(
            player_type_name='neural_network',
            configuration_parameters={
                'learning_rate': 0.001,
                'discount_factor': self.training_config.discount_factor_gamma,
                'exploration_rate': self.training_config.initial_exploration_rate_epsilon
            },
            model_file_path=self.training_config.input_model_file_path,
            creation_timestamp=datetime.now().isoformat()
        )
        return NeuralNetworkCheckersPlayer(player_config)
    
    def _create_training_opponent(self) -> AbstractCheckersPlayer:
        """Create appropriate training opponent based on configuration"""
        opponent_type = self.training_config.training_opponent_type
        
        if opponent_type == "random":
            return create_player_from_configuration('random', {'random_seed': None})
        elif opponent_type == "minimax":
            return create_player_from_configuration('minimax', {'search_depth': 3})
        elif opponent_type == "trainedByFile":
            if not self.training_config.input_model_file_path:
                raise ValueError("trainedByFile opponent requires input_model_file_path")
            return create_player_from_configuration('qlearning', {}, self.training_config.input_model_file_path)
        else:
            raise ValueError(f"Unknown training opponent type: {opponent_type}")
    
    def _execute_training_epoch(self, learning_player: AbstractTrainableCheckersPlayer, 
                               opponent_player: AbstractCheckersPlayer) -> List[CheckersGameResult]:
        """Execute all games for a single training epoch"""
        epoch_game_results = []
        
        for game_number in range(self.training_config.games_per_training_epoch):
            # Alternate starting player
            if game_number % 2 == 0:
                player_one, player_two = learning_player, opponent_player
            # else:
            #     player_one, player_two = opponent_player, learning_player
            
            # Play single game
            game_result = self._play_single_training_game(player_one, player_two)
            epoch_game_results.append(game_result)
            
            # Update learning from game experience
            if isinstance(player_one, AbstractTrainableCheckersPlayer):
                self._update_player_from_game_experience(player_one)
            if isinstance(player_two, AbstractTrainableCheckersPlayer):
                self._update_player_from_game_experience(player_two)
        
        return epoch_game_results
    
    def _play_single_training_game(self, player_one, 
                                  player_two) -> CheckersGameResult:
        """Play a single game between two players"""
        # Reset environment for new game
        current_board_state = self.game_environment.reset_game_to_initial_state()
        
        # Game state tracking for learning
        game_state_sequence = [current_board_state.copy()]
        game_action_sequence = []
        
        current_player_index = 0
        players = [player_one, player_two]
        
        # Main game loop
        while not self.game_environment.is_game_finished:
            current_player = players[current_player_index]
            
            # Get legal actions and player's choice
            legal_actions = self.game_environment.get_all_legal_action_indices()
            
            if not legal_actions:
                break

            chosen_action = current_player.choose_move_from_legal_actions(current_board_state, legal_actions)
            
            # Execute move and get results
            next_state, reward, is_terminal, move_info = self.game_environment.execute_action_by_index(chosen_action)
            
            # Store game progression
            game_state_sequence.append(next_state.copy())
            game_action_sequence.append(chosen_action)
            
            # Update current state and switch players
            current_board_state = next_state
            current_player_index = 1 - current_player_index
        
        # Game completed - create result summary
        game_result = self.game_environment.create_game_result_summary(
            player_one_name=player_one.get_player_type_name(),
            player_two_name=player_two.get_player_type_name(),
            player_one_config=getattr(player_one, 'player_configuration', None),
            player_two_config=getattr(player_two, 'player_configuration', None)
        )
        
        # Notify players of game completion
        player_one_won = (game_result.winning_player_name == 'player_one')
        player_two_won = (game_result.winning_player_name == 'player_two')
        is_draw = (game_result.winning_player_name == 'draw')
        
        player_one.update_after_game_completion({
            'did_win': player_one_won,
            'was_draw': is_draw,
            'game_result': game_result
        })
        
        player_two.update_after_game_completion({
            'did_win': player_two_won,
            'was_draw': is_draw,
            'game_result': game_result
        })
        
        return game_result
    
    def _update_player_from_game_experience(self, trainable_player: AbstractTrainableCheckersPlayer):
        """Update trainable player based on game experience"""
        pass
    
    def _display_epoch_progress(self, epoch_number: int, epoch_stats: Dict, game_results: List[CheckersGameResult]):
        """Display training progress for current epoch"""
        # Calculate epoch-specific statistics
        total_games = len(game_results)
        wins = sum(1 for result in game_results if result.winning_player_name == 'player_one')
        draws = sum(1 for result in game_results if result.winning_player_name == 'draw')
        win_rate = (wins / total_games) * 100 if total_games > 0 else 0
        
        print(f"Epoch {epoch_number:3d}/{self.training_config.number_of_training_epochs:3d} | "
              f"Win Rate: {win_rate:5.1f}% | "
              f"Exploration: {epoch_stats['current_exploration_rate']:.3f} | "
              f"Q-Table Size: {epoch_stats['total_states_in_q_table']:,} | "
              f"Duration: {epoch_stats['epoch_duration_seconds']:.1f}s")
    
    def _generate_final_training_report(self, trained_player: QLearningCheckersPlayer, 
                                      total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive final training report"""
        training_report = {
            'training_session_metadata': {
                'completion_timestamp': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'configuration_used': self.training_config.__dict__,
                'system_version': '1.0.0'
            },
            'final_player_statistics': trained_player.get_player_statistics_summary(),
            'training_progress_statistics': trained_player.get_training_progress_statistics(),
            'epoch_by_epoch_progress': self.epoch_statistics_history,
            'performance_analysis': self._analyze_training_performance(),
            'recommendations': self._generate_training_recommendations(trained_player)
        }
        
        # Save training report
        report_file_path = f"training_report_{int(time.time())}.json"
        with open(report_file_path, 'w') as report_file:
            json.dump(training_report, report_file, indent=2, default=str)
        
        print(f"Detailed training report saved to: {report_file_path}")
        
        return training_report
    
    def _analyze_training_performance(self) -> Dict[str, Any]:
        """Analyze training performance trends"""
        if not self.epoch_statistics_history:
            return {}
        
        # Extract win rates over time
        epoch_win_rates = []
        for epoch_data in self.epoch_statistics_history:
            if 'games_results' in epoch_data:
                games = epoch_data['games_results']
                wins = sum(1 for game in games if game.winning_player_name == 'player_one')
                win_rate = (wins / len(games)) * 100 if games else 0
                epoch_win_rates.append(win_rate)
        
        # Calculate performance trends
        performance_analysis = {
            'initial_win_rate': epoch_win_rates[0] if epoch_win_rates else 0,
            'final_win_rate': epoch_win_rates[-1] if epoch_win_rates else 0,
            'peak_win_rate': max(epoch_win_rates) if epoch_win_rates else 0,
            'average_win_rate': sum(epoch_win_rates) / len(epoch_win_rates) if epoch_win_rates else 0,
            'win_rate_improvement': (epoch_win_rates[-1] - epoch_win_rates[0]) if len(epoch_win_rates) > 1 else 0,
            'training_stability': self._calculate_training_stability(epoch_win_rates)
        }
        
        return performance_analysis
    
    def _calculate_training_stability(self, win_rates: List[float]) -> float:
        """Calculate training stability metric (lower is more stable)"""
        if len(win_rates) < 2:
            return 0.0
        
        # Calculate variance in win rates as stability measure
        mean_win_rate = sum(win_rates) / len(win_rates)
        variance = sum((rate - mean_win_rate) ** 2 for rate in win_rates) / len(win_rates)
        return variance ** 0.5  # Standard deviation
    
    def _generate_training_recommendations(self, trained_player: QLearningCheckersPlayer) -> List[str]:
        """Generate recommendations for improving training"""
        recommendations = []
        
        final_stats = trained_player.get_player_statistics_summary()
        training_stats = trained_player.get_training_progress_statistics()
        
        # Analyze final performance and suggest improvements
        final_win_rate = final_stats.get('win_rate_percentage', 0)
        
        if final_win_rate < 30:
            recommendations.append("Consider increasing learning rate or training duration")
        elif final_win_rate > 80:
            recommendations.append("Excellent performance! Try training against stronger opponents")
        
        if training_stats.get('current_exploration_rate', 0) > 0.1:
            recommendations.append("Consider extending training to reduce exploration rate further")
        
        q_table_size = training_stats.get('q_table_size', 0)
        if q_table_size < 1000:
            recommendations.append("Q-table seems small - consider longer training or more diverse opponents")
        
        return recommendations


class CheckersTournamentSystem:
    """
    Tournament system for comparing different AI approaches
    
    Runs comprehensive tournaments between multiple player types:
    - Round-robin format with configurable game counts
    - Detailed performance analytics and statistical significance
    - Support for all player types (Q-learning, minimax, neural networks, human)
    """
    
    def __init__(self, tournament_configuration: TournamentConfiguration):
        self.tournament_config = tournament_configuration
        self.game_environment = CheckersGameEnvironment(enable_detailed_logging=True)
        self.tournament_results = []
        self.detailed_matchup_statistics = {}
    
    def execute_complete_tournament(self) -> Dict[str, Any]:
        """
        Execute complete tournament with all configured players
        
        Returns:
            Dict: Comprehensive tournament results and analysis
        """
        print("="*80)
        print(f"STARTING TOURNAMENT: {self.tournament_config.tournament_name}")
        print("="*80)
        
        tournament_start_time = time.time()
        
        # Create all participating players
        tournament_players = self._create_tournament_players()
        
        print(f"Tournament Configuration:")
        print(f"  - Participants: {len(tournament_players)} players")
        print(f"  - Games per matchup: {self.tournament_config.games_per_matchup}")
        print(f"  - Randomize starting player: {self.tournament_config.randomize_starting_player}")
        print(f"  - Results file: {self.tournament_config.tournament_output_file_path}")
        print()
        
        # Execute round-robin tournament
        total_matchups = len(tournament_players) * (len(tournament_players) - 1) // 2
        current_matchup = 0
        
        for i in range(len(tournament_players)):
            for j in range(i + 1, len(tournament_players)):
                current_matchup += 1
                player_one = tournament_players[i]
                player_two = tournament_players[j]
                
                print(f"Matchup {current_matchup}/{total_matchups}: "
                      f"{player_one.get_player_type_name()} vs {player_two.get_player_type_name()}")
                
                matchup_results = self._execute_player_matchup(player_one, player_two)
                self._store_matchup_results(player_one, player_two, matchup_results)
        
        # Generate comprehensive tournament report
        tournament_duration = time.time() - tournament_start_time
        tournament_report = self._generate_tournament_report(tournament_players, tournament_duration)
        
        # Save tournament results
        self._save_tournament_results(tournament_report)
        
        print("\n" + "="*80)
        print("TOURNAMENT COMPLETED!")
        print("="*80)
        print(f"Results saved to: {self.tournament_config.tournament_output_file_path}")
        print(f"Tournament duration: {tournament_duration:.2f} seconds")
        
        return tournament_report
    
    def _create_tournament_players(self) -> List[AbstractCheckersPlayer]:
        """Create all players specified in tournament configuration"""
        players = []
        
        for player_spec in self.tournament_config.participating_player_configurations:
            player_type = player_spec['player_type']
            config_path_or_params = player_spec['configuration_path_or_params']
            
            if config_path_or_params == 'default':
                # Use default configuration for player type
                if player_type == 'qlearning':
                    config_params = {'learning_rate': 0.1, 'discount_factor': 0.9}
                elif player_type == 'minimax':
                    config_params = {'search_depth': 5}
                elif player_type == 'random':
                    config_params = {}
                else:
                    config_params = {}
                
                player = create_player_from_configuration(player_type, config_params)
            else:
                # Load from file or parse configuration
                if config_path_or_params.endswith('.json'):
                    player = create_player_from_configuration(player_type, {}, config_path_or_params)
                else:
                    # Parse inline configuration (e.g., "depth5" for minimax)
                    config_params = self._parse_inline_configuration(player_type, config_path_or_params)
                    player = create_player_from_configuration(player_type, config_params)
            
            players.append(player)
        
        return players
    
    def _parse_inline_configuration(self, player_type: str, config_string: str) -> Dict[str, Any]:
        """Parse inline configuration strings like 'depth5' for minimax"""
        if player_type == 'minimax' and config_string.startswith('depth'):
            depth = int(config_string[5:])
            return {'search_depth': depth}
        else:
            return {}
    
    def _execute_player_matchup(self, player_one: AbstractCheckersPlayer, 
                              player_two: AbstractCheckersPlayer) -> List[CheckersGameResult]:
        """Execute all games between two players"""
        matchup_results = []
        
        for game_number in range(self.tournament_config.games_per_matchup):
            # Determine starting player
            if self.tournament_config.randomize_starting_player:
                if game_number % 2 == 0:
                    first_player, second_player = player_one, player_two
                else:
                    first_player, second_player = player_two, player_one
            else:
                first_player, second_player = player_one, player_two
            
            # Play game with optional time limit
            game_result = self._play_tournament_game(first_player, second_player)
            matchup_results.append(game_result)
        
        return matchup_results
    
    def _play_tournament_game(self, player_one: AbstractCheckersPlayer, 
                            player_two: AbstractCheckersPlayer) -> CheckersGameResult:
        """Play a single tournament game between two players"""
        # Reset environment
        current_board_state = self.game_environment.reset_game_to_initial_state()
        
        current_player_index = 0
        players = [player_one, player_two]
        game_start_time = time.time()
        
        # Main game loop with optional time limit
        while not self.game_environment.is_game_finished:
            current_player = players[current_player_index]
            
            # Check time limit
            if (self.tournament_config.time_limit_per_game_seconds and 
                time.time() - game_start_time > self.tournament_config.time_limit_per_game_seconds):
                # Game timed out - declare draw
                self.game_environment.is_game_finished = True
                self.game_environment.winning_player_name = 'draw'
                break
            
            # Get legal actions and player choice
            legal_actions = self.game_environment.get_all_legal_action_indices()
            
            if not legal_actions:
                break
            
            try:
                chosen_action = current_player.choose_move_from_legal_actions(current_board_state, legal_actions)
                next_state, _, is_terminal, _ = self.game_environment.execute_action_by_index(chosen_action)
                current_board_state = next_state
                current_player_index = 1 - current_player_index
                
            except Exception as e:
                print(f"Error in game: {e}")
                # Award win to opponent
                self.game_environment.is_game_finished = True
                self.game_environment.winning_player_name = 'player_two' if current_player_index == 0 else 'player_one'
                break
        
        # Create game result
        return self.game_environment.create_game_result_summary(
            player_one_name=player_one.get_player_type_name(),
            player_two_name=player_two.get_player_type_name()
        )
    
    def _store_matchup_results(self, player_one: AbstractCheckersPlayer, 
                             player_two: AbstractCheckersPlayer, 
                             matchup_results: List[CheckersGameResult]):
        """Store detailed results for a player matchup"""
        matchup_key = f"{player_one.get_player_type_name()}_vs_{player_two.get_player_type_name()}"
        
        # Calculate matchup statistics
        total_games = len(matchup_results)
        player_one_wins = sum(1 for result in matchup_results if result.winning_player_name == 'player_one')
        player_two_wins = sum(1 for result in matchup_results if result.winning_player_name == 'player_two')
        draws = total_games - player_one_wins - player_two_wins
        
        self.detailed_matchup_statistics[matchup_key] = {
            'player_one_type': player_one.get_player_type_name(),
            'player_two_type': player_two.get_player_type_name(),
            'total_games': total_games,
            'player_one_wins': player_one_wins,
            'player_two_wins': player_two_wins,
            'draws': draws,
            'player_one_win_rate': (player_one_wins / total_games) * 100,
            'player_two_win_rate': (player_two_wins / total_games) * 100,
            'draw_rate': (draws / total_games) * 100,
            'average_game_length': sum(result.total_game_moves for result in matchup_results) / total_games,
            'game_results': matchup_results
        }
        
        print(f"  Results: {player_one_wins}-{draws}-{player_two_wins} "
              f"(Win rates: {(player_one_wins/total_games)*100:.1f}% vs {(player_two_wins/total_games)*100:.1f}%)")
    
    def _generate_tournament_report(self, players: List[AbstractCheckersPlayer], 
                                  duration: float) -> Dict[str, Any]:
        """Generate comprehensive tournament report"""
        # Calculate overall standings
        player_standings = self._calculate_player_standings(players)
        
        tournament_report = {
            'tournament_metadata': {
                'tournament_name': self.tournament_config.tournament_name,
                'completion_timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'configuration_used': self.tournament_config.__dict__
            },
            'participant_summary': [
                {
                    'player_type': player.get_player_type_name(),
                    'total_games': player.total_games_played,
                    'performance_stats': player.get_player_statistics_summary()
                }
                for player in players
            ],
            'final_standings': player_standings,
            'detailed_matchup_results': self.detailed_matchup_statistics,
            'tournament_analysis': self._analyze_tournament_results(players),
            'recommendations': self._generate_tournament_recommendations()
        }
        
        return tournament_report
    
    def _calculate_player_standings(self, players: List[AbstractCheckersPlayer]) -> List[Dict[str, Any]]:
        """Calculate final tournament standings"""
        standings = []
        
        for player in players:
            stats = player.get_player_statistics_summary()
            
            # Calculate tournament points (3 for win, 1 for draw, 0 for loss)
            tournament_points = (stats['wins_achieved'] * 3 + stats['draws_reached'] * 1)
            
            standings.append({
                'rank': 0,  # Will be filled after sorting
                'player_type': player.get_player_type_name(),
                'total_points': tournament_points,
                'games_played': stats['total_games_played'],
                'wins': stats['wins_achieved'],
                'draws': stats['draws_reached'],
                'losses': stats['losses_suffered'],
                'win_rate_percentage': stats['win_rate_percentage'],
                'points_per_game': tournament_points / max(1, stats['total_games_played'])
            })
        
        # Sort by points per game, then by total points
        standings.sort(key=lambda x: (x['points_per_game'], x['total_points']), reverse=True)
        
        # Assign ranks
        for i, player_standing in enumerate(standings):
            player_standing['rank'] = i + 1
        
        return standings
    
    def _analyze_tournament_results(self, players: List[AbstractCheckersPlayer]) -> Dict[str, Any]:
        """Analyze tournament results for insights"""
        analysis = {
            'strongest_player': None,
            'most_consistent_player': None,
            'closest_matchups': [],
            'dominant_matchups': [],
            'average_game_length': 0.0,
            'total_games_played': sum(player.total_games_played for player in players) // 2
        }
        
        # Find strongest player (highest win rate)
        best_win_rate = 0
        for player in players:
            stats = player.get_player_statistics_summary()
            if stats['win_rate_percentage'] > best_win_rate:
                best_win_rate = stats['win_rate_percentage']
                analysis['strongest_player'] = {
                    'player_type': player.get_player_type_name(),
                    'win_rate': best_win_rate
                }
        
        # Analyze matchup closeness
        for matchup_key, matchup_data in self.detailed_matchup_statistics.items():
            win_rate_diff = abs(matchup_data['player_one_win_rate'] - matchup_data['player_two_win_rate'])
            
            if win_rate_diff < 10:  # Close matchup
                analysis['closest_matchups'].append({
                    'matchup': matchup_key,
                    'win_rate_difference': win_rate_diff
                })
            elif win_rate_diff > 70:  # Dominant matchup
                analysis['dominant_matchups'].append({
                    'matchup': matchup_key,
                    'win_rate_difference': win_rate_diff
                })
        
        return analysis
    
    def _generate_tournament_recommendations(self) -> List[str]:
        """Generate recommendations based on tournament results"""
        recommendations = []
        
        # Analyze matchup balance
        close_matchups = sum(1 for matchup in self.detailed_matchup_statistics.values() 
                           if abs(matchup['player_one_win_rate'] - matchup['player_two_win_rate']) < 15)
        
        total_matchups = len(self.detailed_matchup_statistics)
        
        if close_matchups / max(1, total_matchups) > 0.7:
            recommendations.append("Tournament shows good balance between player types")
        else:
            recommendations.append("Consider adjusting player configurations for more balanced competition")
        
        return recommendations
    
    def _save_tournament_results(self, tournament_report: Dict[str, Any]):
        """Save tournament results to file"""
        with open(self.tournament_config.tournament_output_file_path, 'w') as results_file:
            json.dump(tournament_report, results_file, indent=2, default=str)


def main():
    """Main application entry point"""
    try:
        # Parse command line arguments
        config_manager = CheckersConfigurationManager()
        args = config_manager.parse_command_line_arguments()
        
        if not args.operation_mode:
            config_manager.command_line_parser.print_help()
            return
        
        # Execute appropriate operation mode
        if args.operation_mode == 'train':
            # Training mode
            training_config = config_manager.create_qlearning_training_config_from_args(args)
            
            # Validate configuration
            validation_errors = config_manager.validate_configuration_parameters(training_config)
            if validation_errors:
                print("Configuration validation errors:")
                for error in validation_errors:
                    print(f"  - {error}")
                return 1
            
            # Execute training
            training_system = CheckersTrainingSystem(training_config)
            training_results = training_system.execute_qlearning_training_session()
            
        elif args.operation_mode == 'tournament':
            # Tournament mode  
            tournament_config = config_manager.create_tournament_config_from_args(args)
            
            # Validate configuration
            validation_errors = config_manager.validate_configuration_parameters(tournament_config)
            if validation_errors:
                print("Configuration validation errors:")
                for error in validation_errors:
                    print(f"  - {error}")
                return 1
            
            # Execute tournament
            tournament_system = CheckersTournamentSystem(tournament_config)
            tournament_results = tournament_system.execute_complete_tournament()
            
        elif args.operation_mode == 'play':
            # Human play mode
            play_config = config_manager.create_play_session_config_from_args(args)
            
            # Create human vs AI play session
            print("Human vs AI play mode not yet implemented in this example")
            print("This would launch the GUI interface for human gameplay")
            
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)