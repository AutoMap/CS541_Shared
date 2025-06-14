#!/usr/bin/env python3
"""
neural_qlearning_agent.py
=========================

Neural network implementation of a Q-learning agent for the checkers environment.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple

from checkers_player_interface import AbstractTrainableCheckersPlayer, CheckersPlayerConfiguration


class DeepQNetwork(nn.Module):
    """
    Neural network for deep Q-learning implementation
    """
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        # Input: 8x8 board state flattened to 64 values
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # Output: Q-values for all possible action indices (max 50 in checkers)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


class NeuralNetworkCheckersPlayer(AbstractTrainableCheckersPlayer):
    def __init__(self, player_configuration: CheckersPlayerConfiguration):
        super().__init__(player_configuration)
        self.model = DeepQNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.memory = []
        self.max_memory_size = 10000
        self.batch_size = 64
        self.training_epochs_completed = 0

        # Add target network
        self.target_model = DeepQNetwork()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_frequency = 5  # Update every 5 games
        self.games_played = 0

        if player_configuration.model_file_path:
            self.load_trained_model_from_file(player_configuration.model_file_path)

    def choose_move_from_legal_actions(self, current_board_state, available_legal_actions):
        # Use a higher epsilon during training
        if random.random() < self.epsilon:
            return random.choice(available_legal_actions)
        
        board_tensor = torch.tensor(current_board_state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(board_tensor).detach().numpy()[0]
        
        # Add small random noise to break ties and increase exploration
        q_values = q_values + np.random.normal(0, 0.05, q_values.shape)
        
        # Prioritize actions with higher Q-values
        filtered_q_values = [(a, q_values[a]) for a in available_legal_actions]
        best_action = max(filtered_q_values, key=lambda x: x[1])[0]
        return best_action

    def update_from_game_experience(self, game_state_sequence, action_sequence, reward_sequence):
        # Create state pairs for Q-learning updates
        for i in range(len(game_state_sequence)-1):
            state = game_state_sequence[i]
            next_state = game_state_sequence[i+1]
            action = action_sequence[i]
            reward = reward_sequence[i]
            
            self.memory.append((state, action, reward, next_state))
        
        # Add final state with terminal flag
        if game_state_sequence:
            self.memory.append((game_state_sequence[-1], action_sequence[-1], 
                            reward_sequence[-1], None))  # None indicates terminal state
        
        # Limit memory size by removing oldest experiences
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
            
        # Training on batches
        if len(self.memory) >= self.batch_size:
            # Calculate priorities based on reward magnitude
            priorities = np.abs([r for _, _, r, _ in self.memory]) + 0.01
            probs = priorities / np.sum(priorities)
            indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
            minibatch = [self.memory[i] for i in indices]
            
            # Process all samples together as tensors
            states = torch.tensor(np.array([s for s, _, _, _ in minibatch]), dtype=torch.float32)
            actions = torch.tensor([a for _, a, _, _ in minibatch], dtype=torch.long)
            rewards = torch.tensor([r for _, _, r, _ in minibatch], dtype=torch.float32)
            
            # Create tensor of next states, replacing terminal states with zeros
            next_states = np.array([n if n is not None else np.zeros_like(s) 
                                  for (s, _, _, n) in minibatch])
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
            
            # Calculate target Q-values in a batch operation
            current_q_values = self.model(states)
            next_q_values = self.target_model(next_states_tensor).detach()
            max_next_q = next_q_values.max(1)[0]
            
            # Create target tensor
            target_q_values = current_q_values.clone()
            for i in range(self.batch_size):
                if minibatch[i][3] is not None:  # Non-terminal state
                    target_q_values[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]
                else:  # Terminal state
                    target_q_values[i, actions[i]] = rewards[i]
            
            # Update model with a single backward pass
            self.optimizer.zero_grad()
            loss = self.loss_fn(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
    
    def update_after_game_completion(self, game_result_data: Dict[str, Any]) -> None:
        """
        Update player after game completion
        """
        self.record_game_outcome(
            did_win=game_result_data.get("did_win", False),
            was_draw=game_result_data.get("was_draw", False),
            additional_game_data=game_result_data
        )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        self.games_played += 1
        if self.games_played % self.target_update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_trained_model_to_file(self, file_path: str) -> None:
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'epochs_completed': self.training_epochs_completed
        }, file_path)

    def load_trained_model_from_file(self, file_path: str) -> None:
        """Load model from file"""
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', 0.9)
            self.training_epochs_completed = checkpoint.get('epochs_completed', 0)

    def get_training_progress_statistics(self) -> Dict[str, Any]:
        """Get training progress statistics"""
        return {
            'model_type': 'neural_network',
            'training_epochs_completed': self.training_epochs_completed,
            'current_epsilon': self.epsilon,
            'batch_size': self.batch_size,
            'memory_size': len(self.memory),
            'total_states_in_q_table': len(self.memory)  # For compatibility with Q-learning reporting
        }

    def complete_training_epoch(self) -> Dict[str, Any]:
        """Complete training epoch and return statistics"""
        self.training_epochs_completed += 1
        self.save_trained_model_to_file("deep_q_model_checkpoint.pt")
        return {
            'training_epochs_completed': self.training_epochs_completed,
            'current_exploration_rate': self.epsilon,
            'total_states_in_memory': len(self.memory),
            'total_states_in_q_table': len(self.memory)  # For compatibility with Q-learning reporting
        }
