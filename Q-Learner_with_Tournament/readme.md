This is the separate program to help train a q-learner.


Checkers AI Training and Competition System

### Operation Modes

The program accepts one of the following **positional arguments** to determine its mode of operation:

| Mode        | Description           |
|-------------|-----------------------|
| `train`     | Train AI players      |
| `tournament`| Run AI tournament     |
| `play`      | Play against AI       |

### Options

| Option         | Description                        |
|----------------|------------------------------------|
| `-h`, `--help` | Show this help message and exit    |


Examples:
  # Quick training with defaults (recommended for first try)
  python checkers_main.py train --quick

  # Custom training parameters
  python checkers_main.py train --learning-rate 0.2 --number-of-epochs 100

  # Run tournament
  python checkers_main.py tournament --tournament-players random:default minimax:depth3

  # Play against trained AI
  python checkers_main.py play --ai-opponent qlearning --ai-model-path qlearning_model.json

py checkers_main.py -help
usage: checkers_main.py [-n TOTAL_GAMES] [--epochs EPOCHS] [--games-per-epoch GAMES_PER_EPOCH] [--train] [-h]
                        player1 player2

English-draughts (8 × 8) tournament / training runner
====================================================

This script lets you pit any two agents against each other
(**random**, **minimax**, or **Q-learning**) **and** drive Q-learning
training runs in an “epochs × games-per-epoch” style.


**PLAYER-SPEC SYNTAX**

Each positional argument (**player1**, **player2**) is a *player-spec* string:

    random
    minimax[depth=6]
    qlearning[
        in=my_start.json,
        out=my_end.json,
        lr=0.15,
        gamma=0.90,
        epsilon=0.7,
        epsilon_decay=0.01,
        decay_interval=5,
        seed=42
    ]

• Inside the brackets is a comma-separated list of **key=value** pairs.
• Whitespace inside the brackets is ignored.
• Values are auto-converted to **int** or **float** whenever possible.


**Q-LEARNING HYPER-PARAMETERS** (inside *qlearning[…]*)

| key (aliases)                | type  | default | description                                     |
|------------------------------|-------|---------|-------------------------------------------------|
| in                           | str   | –       | JSON file to *Load* a Q-table                   |
| out, save_path               | str   | –       | JSON file to *save* the final table             |
| lr, learning_rate, alpha     | float | 0.2     | α – learning-rate                               |
| gamma, discount_factor       | float | 0.95    | γ – discount factor                             |
| epsilon, initial_epsilon     | float | 0.8     | ε₀ – starting exploration rate                  |
| epsilon_decay                | float | 0.02    | Δε – subtracted from ε every interval           |
| decay_interval               | int   | 1       | games between ε updates                         |
| seed                         | int   | –       | RNG seed for reproducible runs                  |



**TOP-LEVEL CLI FLAGS**
### Command Line Arguments

| Argument                  | Description                                 | Default     |
|---------------------------|---------------------------------------------|-------------|
| `--epochs N`              | Train/play for N epochs                     | `1`         |
| `--games-per-epoch M`     | Games inside each epoch                     | `1`         |
| `-n`, `--total-games K`   | Explicit total games; overrides `epochs × games-per-epoch` | — |
| `--train`                 | **Enable Q-updates during play**            | `off`       |
| `-h`, `--help`, `--help`  | Show this full help                         | —           |



**EXAMPLES**

➊ Evaluate – minimax depth-4 vs random, 10 games
   python checkers_main.py minimax[depth=4] random -n 10

➋ **Train** a learner for 1 000 games (10 × 100) and save
   python checkers_main.py        "qlearning[out=q1000.json,lr=0.25,epsilon=0.9]"        random        --epochs 10 --games-per-epoch 100 --train

➌ Continue training with a lower ε
   python checkers_main.py        "qlearning[in=q1000.json,out=q2000.json,epsilon=0.5]"        random        -n 1000 --train

positional arguments:
  player1               first player-spec
  player2               second player-spec

### Training Workflow Overview

**Initialisation**  
`checkers_main.py` builds two players (typically Q-learning vs random), resets the environment, and records the starting board and turn.

**Game Loop**  
On each ply, the current player selects an action index from the environment’s legal-move list.  
Moves are executed using `execute_action_by_index`, which returns the next state and an immediate reward.

**Learning Update**  
After every move (and again at the end of the game), the Q-learner updates the Q-table using the formula:


 Q(s,a)  ←  Q(s,a)  +  α [r  +  γ  max⁡a′Q(s′,a′)  −  Q(s,a)] Q(s,a)
 
The update uses the opponent’s forthcoming legal moves as the bootstrap set and applies scheduled ε-decay.

**Logging**  
For each epoch, the driver prints the number of wins, draws, average game length, and the learner’s current ε.  
This enables quick diagnosis of learning progress or stagnation.


**Key Design Decisions & Extensibility**

State Hashing – Encoding the mover’s colour in the hash prevents value leakage between otherwise identical positions with the wrong turn.


Reward Shaping – A small per-ply reward (+0.01) keeps the table dense enough to learn tactics while still giving ±1.0 for decisive results.  In later development, also added capture and positional rewards


JSON file-based Persistence ensures models stay language-agnostic and version-controlled.

