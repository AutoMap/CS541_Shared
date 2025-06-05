# Checkers AI Training and Competition System

A comprehensive platform for training, evaluating, and comparing different AI approaches to checkers, including Q-learning, neural networks, minimax, and human players.

## 🎯 Project Overview

This system provides a complete framework for:
- **Q-Learning Training**: Traditional reinforcement learning with configurable parameters
- **Neural Network Integration**: Easy interface for students to implement their own neural networks
- **Tournament System**: Compare different AI approaches head-to-head
- **Human vs AI Play**: GUI interface for human gameplay
- **Performance Analysis**: Comprehensive statistics and reporting

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone or download all project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify installation:
   ```bash
   py checkers_main_system.py --help
   ```

### Basic Usage

#### Continue Training from Existing Model
```bash
py checkers_main_system.py train \
    --input-model-path existing_model.json \
    --number-of-epochs 25 \
    --training-opponent trainedByFile
```

#### Run Tournament Between AI Types
```bash
py checkers_main_system.py tournament \
    --tournament-players qlearning:model1.json minimax:depth5 random:default \
    --games-per-matchup 100 \
    --tournament-output tournament_results.json
```

#### Play Against Trained AI
```bash
py checkers_main_system.py play \
    --ai-opponent qlearning \
    --ai-model-path best_model.json \
    --enable-gui
```

## 📁 Project Structure

```
checkers-ai-system/
│
├── checkers_main_system.py              # Main application entry point
├── checkers_game_environment.py         # Core game interface
├── checkers_player_interface.py         # Abstract player interfaces
├── other_checkers_players.py            # Minimax, NeuralNetworks, random, human players
├── checkers_training_configuration.py   # Configuration management
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🧠 For Neural Network Students

### Quick Integration Guide

1. **Copy the template** from `checkers_player_interface.py`:
   ```python
   class ExampleNeuralNetworkPlayerTemplate(AbstractTrainableCheckersPlayer):
   ```

2. **Implement three key methods**:
   ```python
   def choose_move_from_legal_actions(self, current_board_state, available_legal_actions):
       # Your neural network decision logic here
   
   def update_from_game_experience(self, game_state_sequence, action_sequence, reward_sequence):
       # Your neural network training logic here
   
   def save_trained_model_to_file(self, file_path):
       # Save your trained neural network
   ```

3. **Board State Format**:
   - Input: 8x8 numpy array
   - Values: -2 (opponent king), -1 (opponent piece), 0 (empty), 1 (your piece), 2 (your king)
   - Always from current player's perspective

4. **Test your implementation**:
   ```bash
   py checkers_main_system.py tournament \
       --tournament-players neural:your_model.json qlearning:baseline.json
   ```

## 🎯 Training Parameters

### Training Opponents
- **`random`**: Random move selection (good for initial training)
- **`trainedByFile`**: Another trained Q-learning agent (self-play)

## 🏆 Tournament System

### Supported Player Types

- **`neural:model.json`** - Neural network implementation
- **`random:default`** - Random baseline player
- **`human:default`** - Human player (for interactive tournaments)

### Tournament Features
- Round-robin format (every player vs every other player)
- Configurable games per matchup for statistical significance
- Detailed performance analytics and rankings
- Win rates, average game length, and other metrics
- Professional reporting suitable for academic evaluation

## 📊 Output Files

### Training Outputs
- **Model File** (`.json`): Complete trained
- **Training Report** (`.json`): Detailed training progress and analysis
- **Progress Checkpoints**: Periodic saves during training

### Tournament Outputs
- **Tournament Results** (`.json`): Complete tournament statistics and rankings
- **Matchup Details**: Head-to-head results between all player pairs
- **Performance Analysis**: Win rates, consistency metrics, recommendations

### File Format Example
```json
{
  "model_metadata": {
    "player_type": "qlearning",
    "creation_timestamp": "2024-01-15T10:30:00",
    "total_training_epochs": 100,
    "q_table_size": 15420
  },
  "hyperparameters": {
    "learning_rate_alpha": 0.1,
    "discount_factor_gamma": 0.9,
    "exploration_rate_epsilon": 0.05
  },
  "q_value_table": { ... },
  "training_statistics": { ... }
}
```

## 🔧 Advanced Usage

### Self-Play Training
Train against progressively stronger versions of itself:
```bash
# Train initial model
py checkers_main_system.py train --output-model-path generation_1.json

# Train against previous generation
py checkers_main_system.py train \
    --training-opponent trainedByFile \
    --input-model-path generation_1.json \
    --output-model-path generation_2.json
```



### Comprehensive Evaluation
```bash

# Compare in tournament
py checkers_main_system.py tournament \
    --tournament-players \
        nn:model_lr05.json \
        qlearning:model_lr15.json \
        random:default \
    --games-per-matchup 200
```

## 🎓 Academic Features

### Statistical Analysis
- Win rate confidence intervals
- Game length distributions
- Training convergence analysis
- Performance stability metrics

### Reproducible Results
- All configurations saved with results
- Random seed support for deterministic training
- Complete parameter logging
- Timestamp tracking for all experiments

### Professional Reporting
- LaTeX-ready tables and figures
- Comprehensive performance comparisons
- Training progress visualization data
- Publication-quality statistics

## 🐛 Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'pydraughts'**
```bash
# Use the py launcher on Windows
py -m pip install pydraughts
py checkers_main_system.py --help
```

**Permission Errors**
```bash
# Install with user flag
pip install --user pydraughts numpy
```
## 📄 License

Educational use - designed for computer science coursework and research.

