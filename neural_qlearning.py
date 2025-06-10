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
from other_checkers_players import RandomCheckersPlayer



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
    config.exploration_decay_amount = 0.05  # Faster decay
    config.exploration_decay_interval = 1    # Decay every epoch
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

        # Create a separate opponent with fixed weights
        opponent_config = CheckersPlayerConfiguration(
            player_type_name='random',  # Use random player for consistent baseline
            configuration_parameters={}
        )
        
        opponent = RandomCheckersPlayer(opponent_config)

        print("Starting reinforcement training...")
        
        # Use our manual implementation instead of the training system
        training_results = execute_reinforcement_training(agent, opponent, config)
        
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

def execute_reinforcement_training(agent, opponent, config):
    """
    Manually implement reinforcement learning for checkers
    
    Args:
        agent: NeuralNetworkCheckersPlayer agent
        config: Training configuration
    
    Returns:
        Training results dictionary
    """
    print("Starting manual reinforcement learning implementation...")
    
    # Create environment
    env = CheckersGameEnvironment()
    
    # Training stats
    epoch_results = []
    
    for epoch in range(1, config.number_of_training_epochs + 1):
        print(f"Starting epoch {epoch}/{config.number_of_training_epochs}")
        
        # Initialize epoch statistics
        epoch_stats = {
            'epoch_number': epoch,
            'games_won': 0,
            'games_lost': 0,
            'games_drawn': 0,
            'exploration_rate': agent.epsilon
        }
        
        # Play games for this epoch
        for game in range(1, config.games_per_training_epoch + 1):
            if game % 10 == 0:
                print(f"  Playing game {game}/{config.games_per_training_epoch}")
            
            # Reset environment
            env.reset_game_to_initial_state()
            
            # Decide who goes first (alternate)
            agent_plays_first = game % 2 == 1
            
            # Track game states and actions
            game_states = []
            game_actions = []
            
            # Play until game is finished
            current_player = 0  # 0 = first player, 1 = second player
            game_over = False
            
            while not game_over:
                # Get current board state
                board_state = env.get_current_board_state()
                
                # Determine which player's turn it is
                is_agent_turn = (agent_plays_first and current_player == 0) or \
                               (not agent_plays_first and current_player == 1)
                
                # Get legal moves
                legal_moves = list(enumerate(env.pydraughts_board.legal_moves()))
                legal_indices = [idx for idx, _ in legal_moves]
                
                if not legal_moves:
                    # No legal moves, current player loses
                    game_over = True
                    if is_agent_turn:
                        epoch_stats['games_lost'] += 1
                    else:
                        epoch_stats['games_won'] += 1
                    break
                
                # Execute move based on whose turn it is
                if is_agent_turn:
                    game_states.append(board_state)
                    
                    # Choose action
                    action_idx = agent.choose_move_from_legal_actions(board_state, legal_indices)
                    game_actions.append(action_idx)
                    
                    # Find the actual move
                    for idx, move in legal_moves:
                        if idx == action_idx:
                            chosen_move = move
                            break
                else:
                    # Opponent's turn
                    # Flip board perspective for opponent
                    flipped_board = board_state * -1 
                    
                    # Let opponent choose a move
                    move_idx = opponent.choose_move_from_legal_actions(flipped_board, legal_indices)
                    
                    # Find the actual move
                    for idx, move in legal_moves:
                        if idx == move_idx:
                            chosen_move = move
                            break
                
                # Execute the move
                env.pydraughts_board.push(chosen_move)
                
                # Check if game is over
                if env.pydraughts_board.is_over():
                    game_over = True
                    winner = env.pydraughts_board.winner()
                    
                    if winner is None:
                        # Draw
                        epoch_stats['games_drawn'] += 1
                        reward = 0.0
                    elif (winner == 0 and agent_plays_first) or (winner == 1 and not agent_plays_first):
                        # Agent won
                        epoch_stats['games_won'] += 1
                        reward = 1.0
                    else:
                        # Agent lost
                        epoch_stats['games_lost'] += 1
                        reward = -1.0
                else:
                    # Game continues, switch player
                    current_player = 1 - current_player
                    reward = 0.0  # No reward for non-terminal states
            
            # Game complete - create reward sequence
            reward_sequence = [0.0] * (len(game_states) - 1) + [reward]
            
            # Enhance rewards with intermediate feedback (improved rewards)
            enhanced_rewards = []
            for i in range(len(game_states) - 1):
                curr_state = game_states[i]
                next_state = game_states[i+1]
                
                # Base reward from terminal state
                base_reward = reward_sequence[i]
                
                # 1. Reward for capturing opponent pieces (increased from 0.1 to 0.3)
                curr_opp_pieces = np.sum(curr_state < 0)
                next_opp_pieces = np.sum(next_state < 0)
                piece_reward = 0.0
                
                if next_opp_pieces < curr_opp_pieces:
                    # Captured opponent piece(s) - increased reward
                    piece_reward = 0.3 * (curr_opp_pieces - next_opp_pieces)
                
                # 2. Reward for getting kings (new)
                king_reward = 0.0
                if np.sum(next_state == 2) > np.sum(curr_state == 2):
                    # New king created
                    king_reward = 0.5
                
                # 3. Reward for progressing toward king row
                king_progress = 0.0
                curr_pieces_pos = np.where(curr_state == 1)
                next_pieces_pos = np.where(next_state == 1)
                
                if len(curr_pieces_pos[0]) > 0 and len(next_pieces_pos[0]) > 0:
                    # Calculate average distance to king row
                    curr_dist = np.mean(curr_pieces_pos[0])  # Row 0 is king row
                    next_dist = np.mean(next_pieces_pos[0])
                    if next_dist < curr_dist:
                        king_progress = 0.05
                
                # 4. Penalty for losing pieces
                piece_penalty = 0.0
                curr_own_pieces = np.sum(curr_state > 0)
                next_own_pieces = np.sum(next_state > 0)
                if next_own_pieces < curr_own_pieces:
                    piece_penalty = -0.2 * (curr_own_pieces - next_own_pieces)
                
                # 5. Center control reward
                # Center squares are typically 13, 14, 17, 18, 21, 22 in a flattened 8x8 board
                center_squares = [13, 14, 17, 18, 21, 22]
                curr_center_control = 0
                next_center_control = 0
                
                # Count your pieces in center squares
                flat_curr = curr_state.flatten()
                flat_next = next_state.flatten()
                
                for sq in center_squares:
                    if sq < len(flat_curr) and flat_curr[sq] > 0:
                        curr_center_control += 1
                    if sq < len(flat_next) and flat_next[sq] > 0:
                        next_center_control += 1
                
                center_reward = 0.1 * (next_center_control - curr_center_control)
                
                # 6. Defensive positioning reward
                # Reward for keeping pieces at the edges of the board
                edge_rows = [0, 7]  
                edge_cols = [0, 7]
                
                curr_edge_pieces = 0
                next_edge_pieces = 0
                
                # Count pieces on edges
                for row in edge_rows:
                    curr_edge_pieces += np.sum(curr_state[row, :] > 0)
                    next_edge_pieces += np.sum(next_state[row, :] > 0)
                
                for col in edge_cols:
                    curr_edge_pieces += np.sum(curr_state[:, col] > 0)
                    next_edge_pieces += np.sum(next_state[:, col] > 0)
                
                edge_reward = 0.05 * (next_edge_pieces - curr_edge_pieces)
                
                # 7. Mobility reward
                # More legal moves is better
                mobility_reward = 0.0
                # We don't have direct access to number of legal moves here, 
                # but we could approximate by counting empty adjacent squares
                
                # Combine all rewards
                total_reward = base_reward + piece_reward + king_reward + king_progress + \
                              piece_penalty + center_reward + edge_reward
                
                enhanced_rewards.append(total_reward)

            enhanced_rewards.append(reward_sequence[-1])
            
            agent.update_from_game_experience(game_states, game_actions, enhanced_rewards)
            
            agent.update_after_game_completion({
                "did_win": epoch_stats['games_won'] > 0 and game == config.games_per_training_epoch,
                "was_draw": epoch_stats['games_drawn'] > 0 and game == config.games_per_training_epoch
            })
        
        # End of epoch
        print(f"Epoch {epoch} results:")
        print(f"  Win rate: {epoch_stats['games_won'] / config.games_per_training_epoch:.2f}")
        print(f"  Loss rate: {epoch_stats['games_lost'] / config.games_per_training_epoch:.2f}")
        print(f"  Draw rate: {epoch_stats['games_drawn'] / config.games_per_training_epoch:.2f}")
        print(f"  Exploration rate: {agent.epsilon:.4f}")
        
        # Save model periodically
        if epoch % config.save_progress_every_epochs == 0:
            save_path = f"reinforcement_model_epoch_{epoch}.pt"
            agent.save_trained_model_to_file(save_path)
            print(f"  Model saved to {save_path}")
        
        epoch_results.append(epoch_stats)
        
        agent.complete_training_epoch()
        
        # Decay exploration rate if needed
        if epoch % config.exploration_decay_interval == 0:
            agent.epsilon = max(
                config.minimum_exploration_rate, 
                agent.epsilon - config.exploration_decay_amount
            )
    
    # Save final model
    agent.save_trained_model_to_file(config.output_model_file_path)
    
    # Return results
    return {
        'epoch_results': epoch_results,
        'final_model_path': config.output_model_file_path
    }

if __name__ == '__main__':
    train_neural_agent()
