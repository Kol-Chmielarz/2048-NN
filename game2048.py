import random
import numpy as np

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.float32)
        self.score = 0
        self._safe_initialize()

    def _safe_initialize(self):
        """Ensures initial tiles follow the 2/4 combination rule"""
        while True:
            self.board = np.zeros((4, 4), dtype=np.float32)
            self.add_new_tile()
            self.add_new_tile()
            # Count 4s and ensure we have at most one
            if np.count_nonzero(self.board == 4.0) <= 1:
                break

    def add_new_tile(self):
        """Adds a new tile following 90% 2 / 10% 4 probability"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.board[i, j] = 4.0 if random.random() < 0.1 else 2.0

    def compress(self, row):
        """Compress non-zero elements to the front"""
        new_row = row[row != 0]
        return np.pad(new_row, (0, 4 - len(new_row)), mode='constant').astype(np.float32)

    def merge(self, row):
        """Merge adjacent equal tiles with proper 2048 behavior"""
        new_row = []
        i = 0
        while i < len(row):
            if i < len(row) - 1 and row[i] == row[i + 1] and row[i] != 0:
                merged_value = row[i] * 2
                new_row.append(merged_value)
                self.score += merged_value
                i += 2  # Skip the next tile since it's merged
            else:
                new_row.append(row[i])
                i += 1
        return np.pad(new_row, (0, 4 - len(new_row)), mode='constant').astype(np.float32)

    def move_left(self):
        """Process left move with proper merge mechanics"""
        new_board = []
        for row in self.board:
            compressed = self.compress(row)
            merged = self.merge(compressed)
            new_board.append(merged)
        moved = not np.array_equal(self.board, new_board)
        self.board = np.array(new_board, dtype=np.float32)
        return moved

    def move(self, direction, add_tile=True):
        """Execute move with proper rotation handling"""
        rotations = {
            'left': 0,
            'up': 1,
            'right': 2,
            'down': 3
        }
        original_board = self.board.copy()
        self.board = np.rot90(self.board, rotations[direction])
        moved = self.move_left()
        self.board = np.rot90(self.board, -rotations[direction])
        
        if not moved:
            self.board = original_board
            return False
        
        if add_tile and moved:
            self.add_new_tile()
        return True

    def has_2048_tile(self):
        return np.any(self.board == 2048)

    def has_moves_left(self):
        """Check for valid moves following standard 2048 rules"""
        # Check for empty cells first
        if np.any(self.board == 0):
            return True
        
        # Check horizontal and vertical merges
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:
                    return True
                if self.board[j, i] == self.board[j + 1, i]:
                    return True
        return False

    def reset(self):
        """Reset game with proper initialization"""
        self._safe_initialize()
        self.score = 0
        return self.board.copy()

    def get_state(self):
        return self.board.copy()

    def set_state(self, state):
        self.board = state.copy()

    def get_valid_moves(self):
        """Accurate valid move detection"""
        valid_moves = []
        original_board = self.board.copy()
        original_score = self.score
        
        for direction in ['left', 'up', 'right', 'down']:
            self.board = original_board.copy()
            self.score = original_score
            if self.move(direction, add_tile=False):
                valid_moves.append(direction)
        
        self.board = original_board
        self.score = original_score
        return valid_moves

    def is_game_over(self):
        """Proper game termination check"""
        return not self.has_moves_left()