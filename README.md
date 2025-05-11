2048 Deep Reinforcement Learning Agent

This project trains a Deep Q-Network (DQN) agent to play the game 2048 using PyTorch. The agent uses advanced techniques such as:
	•	Double DQN
	•	Dueling network architecture
	•	Prioritized Experience Replay
	•	Feature engineering
	•	Residual CNN blocks
	•	Heuristic exploration with curiosity bonuses
	•	Softmax action selection with temperature scaling
	•	Learning rate scheduling (Cosine Annealing)

 Game Overview

2048 is a sliding tile puzzle game where the player combines like tiles to reach the number 2048. The game ends when no valid moves are left.

This implementation includes a custom environment compatible with the DQN agent.

.
├── game2048.py            # Game environment (reset, move logic, tile merging, etc.)
├── train_2048_dqn.py      # DQN agent with prioritized replay and training loop
├── training_log.csv       # Logs per episode (tile reached, rewards, moves, etc.)
├── 2048_dqn_final.pth     # Saved final model


 Features
	•	State Encoding:
	•	Log-transformed board values
	•	Positional information (row/column)
	•	5 engineered features: empty cell ratio, merge potential, max tile, monotonicity, and smoothness
	•	Neural Network:
	•	Residual CNN blocks over a 3-channel board (values, x-pos, y-pos)
	•	Dueling architecture with separate value and advantage streams
	•	Feature attention mechanism for non-spatial inputs
	•	Training Enhancements:
	•	Prioritized replay buffer using TD-error as priority
	•	Double DQN for target value stabilization
	•	Grad norm clipping
	•	Cosine annealing learning rate scheduler
	•	NaN detection and handling
	•	Exploration:
	•	Softmax-based exploration over valid moves using scaled logits
	•	Heuristic-based action selection with board simulations
	•	Curiosity-based epsilon adjustment
	•	Visualization:
	•	Real-time board rendering with matplotlib


 Training runs for 10,000 episodes by default and logs episode summaries to training_log.csv. Models are saved every 1000 episodes.


 Requirements
	•	Python 3.8+
	•	PyTorch
	•	NumPy
	•	SciPy
	•	matplotlib
