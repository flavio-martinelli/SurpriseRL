"""
Created by Flavio Martinelli at 13:26 07/05/2020
"""

import numpy as np
import numpy.random as rd
import tensorflow as tf
import matplotlib.pyplot as plt

class Maze2D:

    def __init__(self, size=(4, 4), moves_list=None, walls=True, reward_value=1.0, env_reward=None):
        """ The maze keeps a 2D array of cells numbered from 0 to N-1, where N is the total cell number. Such array
        (Maze2d.maze_array) can be shuffled to change the cell order in the maze, the absolute coordinates can be always
        retrieved by calling Maze2d.get_coordinates(position)

        :param size: tuple or scalar indicating size of the maze.
        :param moves_list: np.array containing 2d moves, first dimension selects the moves. Moves must be ordered in
        opposite pairs.
        :param walls: boolean deciding whether agent can move outside of the maze and pop out from the other direction
        (like snake), or not.
        :param reward_val: value of reward.
        :param env_reward: reward structure of the environment (supposed to be constant).
        """

        super(Maze2D, self).__init__()
        assert type(size) is tuple and len(size)==2, f"'size' must be a tuple of length 2, not an {type(size)}"
        assert moves_list is None or type(moves_list) is type(np.array((1,2))), "moves_list must be a np.array"

        self.size = np.array(size)
        self.tot_room_number = np.prod(self.size)
        self.walls = walls
        self.maze_array = np.arange(self.tot_room_number).reshape(self.size)
        self.agent_pos = self.get_random_position()

        # initialize moves
        if moves_list is None:
            self.moves_list = np.array([[0,1], [0,-1], [1,0], [-1,0]]) # right, left, down, up

        # constant reward of the environment
        if env_reward is None:
            self.env_reward = np.zeros_like(self.maze_array, dtype=float)
        else:
            self.env_reward = env_reward

        # reward for reaching the goal
        self.reward_value = reward_value
        self.reward_pos = self.get_random_position()
        while self.reward_pos == self.agent_pos:  # Makes sure reward and agent are not at the same position
            self.reward_pos = self.get_random_position()

        # finalize entire reward array of same structure as maze.array
        self.reward_array = self.env_reward.copy()
        self.reward_array[self._1d_to_2d_idx(self.reward_pos)] = reward_value

        # init useful infos
        self.transition_matrix = self.build_transition_matrix(separate_actions=False)
        self._spike_train_set = False

    def get_random_position(self):
        """ Generate a random position within the maze
        :returns a scalar indicating a random maze position"""
        return rd.randint(self.tot_room_number)

    def _1d_to_2d_idx(self, idx):
        """ :returns 2d index tuple of scalar idx, useful to address the maze_array"""
        return np.where(self.maze_array == idx)

    def get_coordinates(self, idx):
        """ :returns 2d index tuple of scalar idx, useful to address for plotting"""
        return tuple(np.array(np.where(self.maze_array == idx)).squeeze())

    def shuffle(self, separe_actions=False):
        """ Permutation of all cell numbers of the maze, recomputes the transition matrix
        :returns nothing"""

        self.maze_array = np.random.permutation(self.maze_array.reshape(-1)).reshape(self.size)
        self.transition_matrix = self.build_transition_matrix(separate_actions=separe_actions)

    def move_reward(self, next_pos=None):
        """ Moves reward to random position if next_pos is None, else moves reward to selected position. Takes care of
        constant rewards of the environment.

        :param next_pos: cell number in which to place the reward"""

        if next_pos is None:
            next_pos = self.get_random_position()

        self.reward_pos = next_pos
        self.reward_array = self.env_reward.copy()
        self.reward_array[self._1d_to_2d_idx(next_pos)] = self.reward_value

        return next_pos

    def build_transition_matrix(self, separate_actions=False, proba_opposite=0.0, proba_others=0.0):
        """ Builds transition matrix of current cell arrangement of the maze
        :param separate_actions: if True builds a transition matrix for each action, address them in the first dimension
        :param proba_opposite: Probability that opposite action occurs
        :param proba_others: Probability that any other action occur
        """

        proba_selected_action = 1. - proba_opposite - proba_others * (self.moves_list.shape[0]-2)

        if separate_actions is False:
            transition_matrix = np.zeros([self.tot_room_number, self.tot_room_number])
            for p in range(self.tot_room_number):
                for move in self.moves_list:
                    transition_matrix[p, self.step(p, move)[0]] += 1

            # Normalize row by row
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix /= row_sums[:, np.newaxis]

        # case separable actions
        else:
            transition_matrix = np.zeros([len(self.moves_list), self.tot_room_number, self.tot_room_number])
            # Loop through all starting cell positions
            for p in range(self.tot_room_number):
                # Loop through all actions
                for idx_action, action in enumerate(self.moves_list):
                    idx_opposite_action = idx_action + 1 if idx_action % 2 == 0 else idx_action - 1
                    opposite_action = self.moves_list[idx_opposite_action]

                    # update probabilities of selected and opposite moves
                    transition_matrix[idx_action, p, self.step(p, action)[0]] += proba_selected_action
                    transition_matrix[idx_action, p, self.step(p, opposite_action)[0]] += proba_opposite

                    # Loop through all other possible moves that such action can produce (with probabilities)
                    for idx_possible_move, possible_move in enumerate(self.moves_list):
                        if idx_possible_move == idx_action or idx_possible_move == idx_opposite_action:
                            continue
                        else:
                            transition_matrix[idx_action, p, self.step(p, possible_move)[0]] += proba_others

            # Normalize row by row
            row_sums = transition_matrix.sum(axis=(0,2))
            transition_matrix /= row_sums[:, np.newaxis]

        return transition_matrix

    def step(self, start_position, move):
        """ Returns the next cell given the selected position and move, DOES NOT change Maze2D.agent_pos, nor signals
        rewards and coordinates of new position."""

        current_2d_pos = np.array(self._1d_to_2d_idx(start_position)).squeeze()
        new_pos = [current_2d_pos[0] + move[0], current_2d_pos[1] + move[1]]

        # control if new position is legal
        if self.walls is True:
            new_pos = np.clip(new_pos, [0, 0], self.size-1)
        else:  # apply symmetries when indices exceed upper bounds (negative indices work fine)
            if new_pos[0] >= self.size[0]:
                new_pos[0] -= self.size[0]
            if new_pos[1] >= self.size[1]:
                new_pos[1] -= self.size[1]

        return self.maze_array[tuple(new_pos)], tuple(new_pos)

    def perform_action(self, action, proba_opposite=0.05, proba_others=0.05):
        """ Chooses a move according to the action selected and the probability of other action to occur, DOES NOT
        change Maze2d.agent_pos, need to change it directly.
        :param action: Scalar indicating idx of move in the maze.moves_list
        :param proba_opposite: Probability that opposite action occurs
        :param proba_others: Probability that any other action occur
        :returns tuple with new cell position and reward value of such cell
        """

        proba_selected_action = 1. - proba_opposite - proba_others * (self.moves_list.shape[0]-2)
        proba_array = np.ones(self.moves_list.shape[0])*proba_others
        proba_array[action] = proba_selected_action
        idx_opposite = action+1 if action % 2 == 0 else action-1
        proba_array[idx_opposite] = proba_opposite

        sampled = rd.choice(np.arange(len(proba_array)), p=proba_array)
        new_cell, new_pos = self.step(self.agent_pos, self.moves_list[sampled])

        return new_cell, self.reward_array[new_pos]

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

    @staticmethod
    def flatten_spk_train(spk_train):
        """ Reshapes spike train output following the correct convention"""
        return spk_train.reshape(spk_train.shape[0], -1, order='F')

    @staticmethod
    def spk_train_to_tensor(spk_train, dtype=tf.float32):
        """ Returns the flattened spike train in tensor format with first dimension as the batch size, e.g.
        [1, t_steps, n_in] """
        return tf.expand_dims(tf.constant(Maze2D.flatten_spk_train(spk_train), dtype=dtype), axis=0)

    def plot_t_mat(self, ax, t_mat=None):
        if t_mat is None:
            t_mat = self.transition_matrix

        pcol = ax.pcolor(t_mat)
        ax.invert_yaxis()
        return pcol