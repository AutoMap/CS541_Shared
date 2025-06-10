import os
import torch
import numpy as np
import re
from checkers_main_system import CheckersTrainingSystem
from checkers_training_configuration import QLearningTrainingConfiguration
from checkers_game_environment import CheckersGameEnvironment
from neural_qlearning_agent import NeuralNetworkCheckersPlayer, DeepQNetwork
from checkers_player_interface import CheckersPlayerConfiguration
from draughts import Board
from draughts.PDN import PDNReader
import matplotlib.pyplot as plt



def convert_board_to_tensor(board):
    """Convert a draughts board to a tensor representation"""
    env = CheckersGameEnvironment()
    env.pydraughts_board = board
    return env.get_current_board_state().astype(np.float32)

def parse_pdn_to_training_data(pdn_path):
    """Parse a PDN file into training data for supervised pretraining of the Q-network"""
    reader = PDNReader(filename=pdn_path)
    training_data = []

    for game in reader.games:
        board = Board(variant="english") 
        for move in game.moves:
            try:
                # Get board state and legal moves
                state = convert_board_to_tensor(board)
                legal_moves = list(enumerate(board.legal_moves()))

                # Parse move string to list of integers (full move path)
                move_str = str(move)
                if '-' in move_str or 'x' in move_str:
                    move_squares = [int(s) for s in re.findall(r'\d+', move_str)]
                else:
                    continue  # Skip invalid move formats

                # Find matching move in legal moves
                for i, legal_move in legal_moves:
                    legal_path = legal_move.steps_move 

                    if move_squares == legal_path:
                        training_data.append((state, i))
                        found_move = legal_move
                        break
                
                # Execute move
                board.push(found_move)
                
            except Exception as e:
                print(f"Error with move {move}: {e}")
                continue
                
    print(f"Extracted {len(training_data)} training pairs")
    return training_data



def supervised_pretrain(model, data, epochs=100, batch_size=64):
    """Pretrain the model using supervised learning on PDN games"""
    if not data:
        print("No training data available.")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_losses = []
    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0.0
        batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            states, actions = zip(*batch)
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            states_tensor = states_tensor.to(device)
            actions_tensor = actions_tensor.to(device)

            logits = model(states_tensor)
            loss = loss_fn(logits, actions_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        epoch_losses.append(avg_loss)
        print(f"Supervised Epoch {epoch + 1}, Loss: {total_loss / max(1, batches):.4f}")
          # Create and save visualization of training error
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title('Supervised Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('supervised_training_loss.png')
    plt.close()


def train_neural_agent():
    """Main function to train the neural network agent"""
    print("Initializing neural network training...")

    config = QLearningTrainingConfiguration()
    config.player_type_name = 'neural_network'
    config.learning_rate_alpha = 0.001
    config.discount_factor_gamma = 0.95
    config.initial_exploration_rate_epsilon = 1.0
    config.minimum_exploration_rate = 0.05
    config.exploration_decay_amount = 0.01
    config.exploration_decay_interval = 100
    config.number_of_training_epochs = 30
    config.games_per_training_epoch = 100
    config.training_opponent_type = 'random'
    config.output_model_file_path = "trained_deep_q_model.pt"
    config.save_progress_every_epochs = 5
    config.display_training_progress = True

    print("Loading PDN for supervised pretraining...")
    training_data = parse_pdn_to_training_data("OCA_2.1.pdn")

    model = DeepQNetwork()
    supervised_pretrain(model, training_data, epochs=100)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': torch.optim.Adam(model.parameters()).state_dict(),
        'epsilon': 1.0,
        'epochs_completed': 0
    }
    torch.save(checkpoint, "pretrained_model_weights.pt")

    player_config = CheckersPlayerConfiguration(
        player_type_name='neural_network',
        configuration_parameters={
            'learning_rate': config.learning_rate_alpha,
            'discount_factor': config.discount_factor_gamma,
            'exploration_rate': config.initial_exploration_rate_epsilon
        },
        model_file_path="pretrained_model_weights.pt"
    )

    try:
        agent = NeuralNetworkCheckersPlayer(player_config)
        print("Starting reinforcement training...")
        training_system = CheckersTrainingSystem(config)
        training_results = training_system.execute_qlearning_training_session()
        
        # Extract performance metrics from training results
        epochs = []
        win_rates = []
        loss_rates = []
        draw_rates = []
        
        for epoch_results in training_results['epoch_results']:
            epochs.append(epoch_results['epoch_number'])
            wins = epoch_results['games_won']
            losses = epoch_results['games_lost']
            draws = epoch_results['games_drawn']
            total_games = wins + losses + draws
            
            win_rates.append(wins / total_games if total_games > 0 else 0)
            loss_rates.append(losses / total_games if total_games > 0 else 0)
            draw_rates.append(draws / total_games if total_games > 0 else 0)
        
        # Plot performance metrics
        plt.figure(figsize=(12, 8))
        
        # Win/Loss/Draw rates
        plt.subplot(2, 1, 1)
        plt.plot(epochs, win_rates, 'g-', label='Win Rate')
        plt.plot(epochs, loss_rates, 'r-', label='Loss Rate')
        plt.plot(epochs, draw_rates, 'b-', label='Draw Rate')
        plt.title('Reinforcement Learning Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Rate')
        plt.legend()
        plt.grid(True)
        
        # Exploration rate
        plt.subplot(2, 1, 2)
        exploration_rates = [epoch_results.get('exploration_rate', 0) for epoch_results in training_results['epoch_results']]
        plt.plot(epochs, exploration_rates, 'm-')
        plt.title('Exploration Rate Decay')
        plt.xlabel('Epoch')
        plt.ylabel('Exploration Rate (ε)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('reinforcement_training_metrics.png')
        plt.close()
        
        print("Reinforcement training visualization saved to reinforcement_training_metrics.png")
        print("Training complete. Final model saved to:", config.output_model_file_path)
    except Exception as e:
        print(f"Error during reinforcement training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    train_neural_agent()
