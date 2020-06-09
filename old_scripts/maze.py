"""
Created by Flavio Martinelli at 18:25 10/03/2020
"""
import numpy as np
import numpy.random as rd
import tensorflow as tf


class Maze:

    def __init__(self, size=(4, 4)):
        """ The maze

        :param size: tuple or scalar indicating size of the maze
        :param current_pos: this describes the current position in the maze,
                            it is a np.array (e.g. [2,5], [7], [1,0,0])
        """
        super(Maze, self).__init__()
        assert type(size) is tuple or type(size) is int, f'size must be a tuple or an int, not an {type(size)}'

        if type(size) is not int:
            self.size = np.array(size)
        else:
            self.size = np.array([size])

        self.maze_dimensionality = len(self.size)
        self.tot_room_number = np.prod(self.size)
        self._spike_train_set = False

        self.current_pos = self.get_random_position()

        self.shuffler = np.arange(0, np.prod(size))

    def get_random_position(self):
        """ Generate a random position within the maze

        :returns a np.array indicating the position in the maze, if 1D it returns e.g. [2]"""

        pos = []
        for i in range(self.maze_dimensionality):
            pos.append(np.random.randint(0, self.size[i]))

        return np.array(pos)

    def set_position(self, pos):
        """ Set a specific position for the maze, not optimal because it runs some checks."""

        assert type(pos) is tuple or type(pos) is int, f'size must be a tuple or an int, not an {type(pos)}'
        if type(pos) is not int:
            pos = np.array(pos)
        else:
            pos = np.array([pos])

        assert len(pos.shape) == self.maze_dimensionality, f'pos is not of the same dimensions as the maze'

        self.current_pos = np.array(pos)

    def set_spike_train_params(self, t_steps, high_freq_p=1., low_freq_p=0.):
        """ Set up spike train statistics, need to do it as initialization of the maze.
        :param t_steps: number of total time steps of the maze
        :param high_freq_p: spike frequency probability of active maze cell
        :param low_freq_p: spike frequency probability of inactive maze cells
        """
        # assert stuff
        assert high_freq_p <= 1.0, f"high_freq_p={high_freq_p} cannot be greater than 1.0"
        assert low_freq_p <= 1.0, f"low_freq_p={low_freq_p} cannot be greater than 1.0"
        assert high_freq_p >= 0.0, f"high_freq_p={high_freq_p} cannot be negative"
        assert low_freq_p >= 0.0, f"low_freq_p={low_freq_p} cannot be negative"
        assert type(t_steps) is int, f"t_steps must be an integer, not a {type(t_steps)}"

        self._t_steps = t_steps
        self._high_freq_p = high_freq_p
        self._low_freq_p = low_freq_p

        self._spike_train_set = True

    def shuffle_maze(self):
        rd.shuffle(self.shuffler)
        idx = np.arange(np.prod(self.size[0]))
        self.transition_matrix = self.transition_matrix[self.shuffler[idx], self.shuffler[idx]]
        return self.transition_matrix

    def switch_indices(self, position):
        new_idx = self.shuffler[self._idx_to_1d(position)]
        delta = np.zeros(np.prod(self.size))
        delta[new_idx] = 1
        return np.argwhere(np.reshape(delta, self.size, order='F') == 1)[0]

    def build_transition_matrix(self, dims=2, symmetric=True, no_duplicates=True):
        """ Generate list of possible moves and their corresponding operator to apply to the current cell position. Plus
            defines and returns the transition matrix of the maze
        :param dims: how many dimensions of movement are allowed (2 symmetric dims will produce 4 possible moves,
                     2 non symmetric dims will generate only 2 moves. First two symmetric dimensions are along x and y,
                     from the third the next room is picked with a fixed random rule (e.g. move 2 up and 1 left)
        :param symmetric: symmetric moves correspond to left-right, up-down, +random and -random
        :param no_duplicates: allows for duplicates in the move list
        :returns transition matrix of the maze
        """

        self.move_symmetric = symmetric
        self.move_dims = dims if not symmetric else dims*2
        self.move_dict = self._generate_possible_moves(dims, symmetric, no_duplicates)

        self.transition_matrix = np.zeros([self.tot_room_number, self.tot_room_number])

        for room_pos, _ in np.ndenumerate(np.zeros(self.size)):  # Loop through all rooms
            for step in self.move_dict.values():  # Loop through all possible moves and record landing room
                next_room_pos = self.move(np.array(room_pos), step)
                self.transition_matrix[self._idx_to_1d(room_pos), self._idx_to_1d(next_room_pos)] += 1
                if self.move_symmetric:
                    self.transition_matrix[self._idx_to_1d(next_room_pos), self._idx_to_1d(room_pos)] += 1

        # Normalize
        if symmetric:
            self.transition_matrix /= 2*self.move_dims
        else:
            self.transition_matrix /= self.move_dims

        return self.transition_matrix

    def _generate_possible_moves(self, dims=2, symmetric=True, no_duplicates=True):
        """ Generate list of possible moves and their corresponding operator to apply to the current cell position
        :param dims: how many dimensions of movement are allowed (2 symmetric dims will produce 4 possible moves,
                     2 non symmetric dims will generate only 2 moves. First two symmetric dimensions are along x and y,
                     from the third the next room is picked with a fixed random rule (e.g. move 2 up and 1 left)
        :param symmetric: symmetric moves correspond to left-right, up-down, +random and -random
        :param no_duplicates: allows for duplicates in the move list
        :returns dictionary of possible moves and their corresponding operator
        """
        if symmetric and no_duplicates:
            # -1 because move 0,0 does not exist
            assert dims <= np.prod(self.size) - 1, \
                f"dims ({dims}) exceeds max possible number of moves for maze size {self.size}"
        elif no_duplicates:
            assert dims <= 2 * np.prod(self.size) - 1, \
                f"dims ({dims}) exceeds max possible number of moves for maze size {self.size}"

        moves = []

        for d in range(dims):
            # Move along only one dimension for the first 2 moves
            if symmetric and d < 2 and not (d == 1 and self.maze_dimensionality == 1):

                # Account for single 1D maze
                if self.maze_dimensionality > 1:
                    move = np.zeros(self.maze_dimensionality, dtype=int)
                else:
                    move = np.zeros([self.maze_dimensionality], dtype=int)

                move[d] = 1
                moves.append(move.copy())
                move[d] = self.size[d] - 1  # avoid negative moves to ease check of duplicates later
                moves.append(move.copy())

            else:  # Move randomly (teleport) for d > 2
                move = rd.randint(low=0, high=self.size, size=self.maze_dimensionality)

                # Make sure that the randomly generated move is not already a possible move or a [0,0] move
                while no_duplicates and (any(np.array_equal(m, move) for m in moves) or np.count_nonzero(move) == 0):
                    move = rd.randint(low=0, high=self.size, size=self.maze_dimensionality)

                moves.append(move)
                if symmetric:
                    moves.append(-move)

        return dict(list(enumerate(moves)))

    def apply_transition(self):
        """ Moves the current position according to the possible moves
        :returns the new current position
        """
        step = self.move_dict[rd.randint(low=0, high=len(self.move_dict.keys()))]
        self.current_pos = self.move(self.current_pos, step)
        return self.switch_indices(self.current_pos), step

    def move(self, init_room, step):
        return np.mod(init_room + step, self.size)

    def generate_spike_train(self, pos):
        """ Generates a spike train matrix for the entire maze, the spike frequency is higher for the current position
            in the maze.
        :param pos: tuple of indices indicating the position in the maze,
                    NB: if pos is scalar it must be a single element list e.g. [3]

        :returns spike_matrix of dimensions [t_steps, *maze_size]
        """

        if not self._spike_train_set:
            self._t_steps = 100
            self._high_freq_p = 1.
            self._low_freq_p = 0.

        spike_matrix_shape = np.hstack([self.size, self._t_steps]).reshape(-1)
        spike_matrix = np.array(rd.uniform(0., 1., size=spike_matrix_shape) < self._low_freq_p, dtype=float)
        spike_matrix[tuple(pos)] = np.array(rd.uniform(0., 1., size=self._t_steps) < self._high_freq_p, dtype=float)

        return np.moveaxis(spike_matrix, -1, 0)

    def _idx_to_1d(self, idx):
        """ Transform a room index in the 1D flattened index """
        return np.sum(idx * np.hstack([1, np.cumprod(self.size[0:-1])]))

    @staticmethod
    def flatten_spk_train(spk_train):
        """ Reshapes spike train output following the correct convention"""
        return spk_train.reshape(spk_train.shape[0], -1, order='F')

    @staticmethod
    def spk_train_to_tensor(spk_train, dtype=tf.float32):
        """ Returns the flattened spike train in tensor format with first dimension as the batch size, e.g.
        [1, t_steps, n_in] """
        return tf.expand_dims(tf.constant(Maze.flatten_spk_train(spk_train), dtype=dtype), axis=0)