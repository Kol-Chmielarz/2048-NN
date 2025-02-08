import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from game2048 import Game2048

# Clear GPU memory, select device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
RENDER_EVERY = 50   # Render every N episodes (disable during training if needed)
BATCH_SIZE = 1024   # Consider reducing if transitions are sparse
GAMMA = 0.9995
EPSILON_START = 1.0  
EPSILON_MIN = 0.05  
EPSILON_DECAY = 0.997
LEARNING_RATE = 0.0005
MEMORY_SIZE = 1500000
TARGET_UPDATE_FREQ = 500   # Update target network every N training steps
ALPHA = 0.7
BETA_START = 0.4
BETA_FRAMES = 500000

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def display_board(state):
    """Visualize the 2048 game board with matplotlib"""
    plt.ion()
    tile_colors = {
        0: [0.9, 0.9, 0.9], 2: [1.0, 1.0, 1.0],
        4: [1.0, 1.0, 0.5], 8: [1.0, 0.6, 0.2],
        16: [1.0, 0.0, 0.0], 32: [0.0, 1.0, 0.0],
        64: [0.0, 0.7, 1.0], 128: [0.5, 0.3, 0.1],
        256: [0.6, 0.0, 0.8], 512: [0.5, 0.5, 0.5],
        1024: [0.0, 0.4, 0.0], 2048: [0.0, 0.0, 0.6]
    }

    if not hasattr(display_board, 'fig'):
        display_board.fig, display_board.ax = plt.subplots()
        display_board.ax.set_title("2048 Game Board")
    else:
        display_board.ax.clear()

    board = state.reshape(4, 4) if state.size == 16 else state
    color_grid = np.zeros((4, 4, 3))
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            color = tile_colors.get(val, [0.0, 0.0, 0.0])
            color_grid[i, j] = color

    display_board.ax.imshow(color_grid, interpolation='nearest')
    for i in range(4):
        for j in range(4):
            val = board[i, j]
            if val > 0:
                brightness = np.mean(color_grid[i, j])
                text_color = 'black' if brightness > 0.5 else 'white'
                display_board.ax.text(j, i, str(val), ha='center', va='center', 
                                       color=text_color, fontweight='bold', fontsize=14)
    display_board.ax.set_xticks([])
    display_board.ax.set_yticks([])
    display_board.fig.canvas.draw()
    plt.pause(0.1)
    display_board.fig.show()

# ----------------- Prioritized Replay Buffer -----------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = Transition(state, action, reward, next_state, done)
        self.priorities[self.pos] = (self.max_priority ** self.alpha) + 1e-8
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=BETA_START):
        if len(self.buffer) == 0:
            return [], [], []
        priorities = self.priorities[:len(self.buffer)]
        if np.any(np.isnan(priorities)):
            priorities = np.nan_to_num(priorities, nan=1e-5)
        sum_priorities = np.sum(priorities)
        if sum_priorities <= 0:
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = priorities / sum_priorities
        if np.any(np.isnan(probs)):
            probs = np.ones_like(priorities) / len(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = torch.FloatTensor(weights / weights.max()).to(device) if len(weights) > 0 else torch.tensor([])
        samples = [self.buffer[idx] for idx in indices]
        return samples, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority ** self.alpha) + 1e-8
        current_max = np.max(self.priorities[:len(self.buffer)])
        self.max_priority = max(self.max_priority, current_max)

    def __len__(self):
        return len(self.buffer)

# ----------------- Enhanced DQN Network -----------------
class EnhancedDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # The board input now has 3 channels: processed board, pos_x, pos_y.
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            ResidualBlock(256),
            nn.Flatten()
        )
        # Feature attention on the extra 5 features.
        self.feature_attention = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
        # After convolution, output is 256*4*4 = 4096; concatenated with 5 extra features.
        self.value_stream = nn.Sequential(
            nn.Linear(4096 + 5, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(4096 + 5, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        # x is (batch, 53): first 48 = board and positional info, next 5 = extra features.
        board = x[:, :48].view(-1, 3, 4, 4)
        features = x[:, 48:53]
        conv_out = self.conv(board)  # shape: (batch, 4096)
        attn_weights = self.feature_attention(features)
        weighted_features = features * attn_weights  # elementwise multiplication
        combined = torch.cat([conv_out, weighted_features], dim=1)  # (batch, 4096+5)
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# ----------------- Residual Block -----------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.conv(x)

# ----------------- DQN Trainer -----------------
class DQNTrainer:
    def __init__(self, env, num_episodes=10000, batch_size=BATCH_SIZE, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY,
                 lr=LEARNING_RATE, memory_size=MEMORY_SIZE):
        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.beta = BETA_START
        self.device = device
        self.episode_memory = deque(maxlen=100)
        self.tau = 0.005
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.model = EnhancedDQN().to(self.device)
        self.target_model = EnhancedDQN().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_counter = 0
        self.episode_grad_norms = []
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # We now schedule once per episode rather than on each training step.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=500,
            T_mult=1,
            eta_min=1e-6
        )
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.action_to_index = {"left":0, "up":1, "right":2, "down":3}
        self.index_to_action = {v: k for k, v in self.action_to_index.items()}
        self.success_count = 0
        self.episode_losses = []
        self.episode_rewards = []
        self.action_counts = {"left":0, "right":0, "up":0, "down":0}
        self.episode_tiles = set()

    def _preprocess_state(self, state):
        """
        Process a 4x4 board:
          - Apply log2 (using 1 for zeros so log2(1)=0), then normalize by dividing by 16.
          - Create two positional channels (normalized row and column indices).
          - Compute 5 extra features.
        Returns a flat numpy vector of length 53.
        """
        state = state.astype(np.float32)
        board_processed = np.log2(np.where(state == 0, 1, state)) / 16.0
        pos_x = np.tile(np.linspace(0, 1, 4).reshape(4, 1), (1, 4))
        pos_y = np.tile(np.linspace(0, 1, 4), (4, 1))
        board_stacked = np.stack([board_processed, pos_x, pos_y], axis=0)
        board_flat = board_stacked.flatten()  # 48 values
        empty_cells = np.sum(state == 0) / 16.0
        merge_pot = self._merge_potential(state) / 24.0
        max_tile = np.max(state) / 4096.0
        monotonicity = self._monotonicity_score(state)
        smoothness = self._smoothness(state)
        features = np.array([empty_cells, merge_pot, max_tile, monotonicity, smoothness], dtype=np.float32)
        return np.concatenate([board_flat, features]).astype(np.float32)

    def _merge_potential(self, board):
        merge_count = 0
        for i in range(4):
            for j in range(3):
                if board[i, j] == board[i, j+1] and board[i, j] > 0:
                    merge_count += 1
                if board[j, i] == board[j+1, i] and board[j, i] > 0:
                    merge_count += 1
        return merge_count

    def _monotonicity_score(self, board):
        row_mono = sum(1 for row in board if np.all(np.diff(row) >= 0) or np.all(np.diff(row) <= 0))
        col_mono = sum(1 for col in board.T if np.all(np.diff(col) >= 0) or np.all(np.diff(col) <= 0))
        return (row_mono + col_mono) / 8.0

    def _smoothness(self, board):
        smooth = 0.0
        for i in range(4):
            for j in range(3):
                if board[i, j] > 0 and board[i, j+1] > 0:
                    smooth += abs(np.log2(board[i, j]) - np.log2(board[i, j+1]))
            for j in range(3):
                if board[j, i] > 0 and board[j+1, i] > 0:
                    smooth += abs(np.log2(board[j, i]) - np.log2(board[j+1, i]))
        return smooth / 24.0

    def _get_reward(self, old_state, new_state):
        """
        Compute reward based on improvement:
          - A bonus for increasing the max tile,
          - Additional bonus if passing thresholds (1024, 2048),
          - Reward based on merged value, board order (positional bonus),
          - Bonus for empty cells and a constant if game not over.
        Note: We remove the division by 10 to amplify the reward signals.
        """
        reward = 0.0
        new_max = np.max(new_state)
        old_max = np.max(old_state)
        if new_max > old_max:
            reward += 2 ** (np.log2(new_max) - 6)
            if new_max >= 1024 and old_max < 1024:
                reward += 500
            if new_max >= 2048 and old_max < 2048:
                reward += 5000
        merged_value = np.sum((new_state - old_state)[new_state > old_state])
        reward += merged_value * (np.log2(new_max + 1) / 11)
        max_pos = np.unravel_index(np.argmax(new_state), (4, 4))
        reward += (3 - max_pos[0])**2 + (3 - max_pos[1])**2
        empty_cells = np.sum(new_state == 0)
        reward += np.sqrt(empty_cells) * 0.5
        if not self.env.is_game_over():
            reward += 5.0
        return reward  # Removed division by 10 for a stronger learning signal

    def _update_target_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, processed_state):
        valid_moves = self.env.get_valid_moves()
        state = self.env.get_state()
        current_max = np.max(state)
        base_epsilon = max(self.epsilon, 0.2 * (1 - current_max / 4096))
        temperature = 1.0 / (np.log2(current_max + 1) + 1e-5)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
            pred_reward = self.model(state_tensor).max().item()
            curiosity_bonus = 0.1 * (5 - pred_reward)
        if random.random() < base_epsilon + curiosity_bonus:
            # Choose a random move (using logits from the model as bias if available)
            with torch.no_grad():
                logits = self.model(state_tensor)[0].cpu().numpy()
            valid_indices = [i for i in range(4) if self.index_to_action[i] in valid_moves]
            if not valid_indices:
                return random.choice(["left", "right", "up", "down"])
            valid_logits = np.array([logits[i] for i in valid_indices], dtype=np.float32)
            scaled_logits = valid_logits / temperature
            scaled_logits = np.clip(scaled_logits, -50, 50)
            probs = softmax(scaled_logits)
            if not np.all(np.isfinite(probs)) or np.isclose(np.sum(probs), 0):
                probs = np.ones_like(probs) / len(probs)
            return random.choices([self.index_to_action[i] for i in valid_indices], weights=probs, k=1)[0]
        else:
            # Use a simple heuristic simulation over valid moves:
            best_move = None
            best_score = -np.inf
            for move in valid_moves:
                temp_env = Game2048()
                temp_env.set_state(state.copy())
                if temp_env.move(move):
                    sim_state = temp_env.get_state()
                    score = (self._merge_potential(sim_state) * 2 +
                             np.sum(sim_state == 0) * 0.5 +
                             np.max(sim_state) * 0.1)
                    if score > best_score:
                        best_score = score
                        best_move = move
            return best_move if best_move is not None else random.choice(valid_moves)

    def train_step(self):
        transitions, weights, indices = self.memory.sample(self.batch_size, self.beta)
        if not transitions:
            return False
        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.tensor([self.action_to_index[a] for a in batch.action], dtype=torch.long).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        current_q = self.model(states)

        # ------------------ Double DQN update ------------------
        with torch.no_grad():
            # Use the online network to choose next action...
            next_actions = self.model(next_states).argmax(1, keepdim=True)
            # ...and use the target network to evaluate that action.
            next_q = self.target_model(next_states).gather(1, next_actions).squeeze(1)
        
        targets = current_q.clone()
        batch_idx = torch.arange(len(transitions), dtype=torch.long)
        targets[batch_idx, actions] = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.criterion(current_q, targets).mean(dim=1)
        prios = loss.detach() + 1e-5
        loss = (loss * weights).mean()
        if torch.isnan(loss):
            print("NaN loss detected, skipping update")
            return True
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        if torch.isnan(grad_norm):
            print("NaN gradients detected, skipping update")
            return True
        self.optimizer.step()
        self.memory.update_priorities(indices, prios.cpu().numpy())
        self.target_update_counter += 1
        if self.target_update_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        self.episode_losses.append(loss.item())
        self.episode_grad_norms.append(grad_norm.item())
        self.beta = min(1.0, BETA_START + (1.0 - BETA_START) * (len(self.memory) / BETA_FRAMES))
        return True

    def train(self):
        with open("training_log.csv", "w") as log_file:
            log_file.write("Episode,MaxTile,Successes,MovesTo2048,Epsilon,AvgLoss,AvgReward,TileCount,LR,2048Achieved\n")
            for episode in range(self.num_episodes):
                state = self.env.reset()
                processed_state = self._preprocess_state(state)
                done = False
                total_reward = 0
                moves = 0
                max_tile = 2
                render_episode = (episode + 1) % RENDER_EVERY == 0
                self.episode_tiles = set()
                achieved_2048 = False
                while not done:
                    moves += 1
                    prev_state = state.copy()
                    action = self.select_action(processed_state)
                    if not self.env.move(action):
                        continue
                    new_state = self.env.get_state()
                    if render_episode:
                        display_board(new_state)
                    processed_next = self._preprocess_state(new_state)
                    done = self.env.is_game_over()
                    new_max = np.max(new_state)
                    if new_max > max_tile:
                        max_tile = new_max
                        for tile in [64, 128, 256, 512, 1024, 2048]:
                            if new_max >= tile and tile not in self.episode_tiles:
                                self.episode_tiles.add(tile)
                    reward = self._get_reward(prev_state, new_state)
                    total_reward += reward
                    self.memory.add(processed_state, action, reward, processed_next, done)
                    if self.train_step():
                        pass  # Training step performed
                    if new_max >= 2048 and not done:
                        self.success_count += 1
                        achieved_2048 = True
                        done = True
                    processed_state = processed_next
                    state = new_state
                    self.episode_rewards.append(total_reward)
                self.episode_memory.append(max_tile)
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                current_lr = self.optimizer.param_groups[0]['lr']
                avg_loss = np.mean(self.episode_losses[-100:]) if self.episode_losses else 0
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                tile_count = {
                    64: int(64 in self.episode_tiles),
                    128: int(128 in self.episode_tiles),
                    256: int(256 in self.episode_tiles),
                    512: int(512 in self.episode_tiles),
                    1024: int(1024 in self.episode_tiles),
                    2048: int(2048 in self.episode_tiles)
                }
                log_line = (
                    f"{episode+1},{max_tile},{self.success_count},"
                    f"{moves if achieved_2048 else 0},{self.epsilon:.4f},"
                    f"{avg_loss:.4f},{avg_reward:.2f},\"{tile_count}\","
                    f"{current_lr:.2e},{'Yes' if achieved_2048 else 'No'}"
                )
                log_file.write(log_line + "\n")
                if (episode + 1) % 100 == 0:
                    print(
                        f"Ep {episode+1} | Max: {max_tile} | ε: {self.epsilon:.3f} | "
                        f"Avg Reward: {avg_reward:.1f} | 2048s: {self.success_count} | "
                        f"LR: {current_lr:.2e}"
                    )
                # Step the scheduler once per episode:
                self.scheduler.step(episode)
                if (episode + 1) % 1000 == 0:
                    torch.save(self.model.state_dict(), f"2048_dqn_{episode+1}.pth")
                    if self.success_count > 0:
                        torch.save(self.model.state_dict(), f"2048_dqn_success_{episode+1}_{self.success_count}.pth")
            torch.save(self.model.state_dict(), "2048_dqn_final.pth")
            print(f"\nTraining completed. Reached 2048 {self.success_count} times. Final model saved.")

if __name__ == "__main__":
    try:
        env = Game2048()
        trainer = DQNTrainer(env, num_episodes=10000)
        trainer.train()
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()