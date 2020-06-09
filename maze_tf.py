"""
Created by Flavio Martinelli at 12:18 18/05/2020
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import numpy.random as rd

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class Maze2D(py_environment.PyEnvironment):
    def __init__(self, size=(4, 4),
                 gamma=1.0,
                 moves_list=None,
                 walls=True,
                 reward_value=1.0,
                 env_reward=None,
                 proba_opposite=0.0,
                 proba_others=0.0):
        """ The maze keeps a 2D array of cells numbered from 0 to N-1, where N is the total cell number. Such array
        (Maze2d.maze_array) can be shuffled to change the cell order in the maze, the absolute coordinates can be always
        retrieved by calling Maze2d.get_coordinates(position)

        :param size: tuple or scalar indicating size of the maze.
        :param gamma: discount value for future rewards.
        :param moves_list: np.array containing 2d moves, first dimension selects the moves. Moves must be ordered in
        opposite pairs.
        :param walls: boolean deciding whether agent can move outside of the maze and pop out from the other direction
        (like snake), or not.
        :param reward_val: value of reward.
        :param env_reward: reward structure of the environment (supposed to be constant).
        :param proba_opposite: Probability that opposite action occurs.
        :param proba_others: Probability that any other action occur.
        """
        assert type(size) is tuple and len(size)==2, f"'size' must be a tuple of length 2, not an {type(size)}"
        assert moves_list is None or type(moves_list) is type(np.array((1,2))), "moves_list must be a np.array"

        self.size = np.array(size)
        self.gamma = gamma
        self.tot_room_number = np.prod(self.size)
        self.walls = walls
        self.maze_array = np.arange(self.tot_room_number).reshape(self.size)
        self._state = self.get_random_position()
        self.reset_state = None

        # initialize moves
        if moves_list is None:
            self.moves_list = np.array([[0,1], [0,-1], [1,0], [-1,0]]) # right, left, down, up

        self.proba_opposite = proba_opposite
        self.proba_others = proba_others
        self.proba_selected_action = 1. - proba_opposite - proba_others * (self.moves_list.shape[0]-2)

        # constant reward of the environment
        if env_reward is None:
            self.env_reward = np.zeros_like(self.maze_array, dtype=float)
        else:
            self.env_reward = env_reward

        # reward for reaching the goal
        self.reward_value = reward_value
        self.reward_pos = self.get_random_position()
        while self.reward_pos == self._state:  # Makes sure reward and agent are not at the same position
            self.reward_pos = self.get_random_position()

        # finalize entire reward array of same structure as maze.array
        self.reward_array = self.env_reward.copy()
        self.reward_array[self._1d_to_2d_idx(self.reward_pos)] = reward_value

        # init useful infos
        self.transition_matrix = self.build_transition_matrix(separate_actions=False)
        self._spike_train_set = False

        # initialize params needed for tf wrapper
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=len(self.moves_list)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(1,), dtype=np.int32, minimum=0, maximum=self.tot_room_number-1, name='observation')

        self._episode_ended = False

    def action_spec(self):
        """py_environment.PyEnvironment function that returns max and min specs for the action values """
        return self._action_spec

    def observation_spec(self):
        """py_environment.PyEnvironment function that returns max and min specs for the observation values """
        return self._observation_spec

    def _reset(self):
        """py_environment.PyEnvironment function that implements a reset"""
        if self.reset_state is None:
            self._state = self.get_random_position()
        else:
            self._state = self.reset_state
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def get_random_position(self):
        """ Generate a random position within the maze
        :returns a scalar indicating a random maze position"""
        return rd.randint(self.tot_room_number)

    def _step(self, action):
        """py_environment.PyEnvironment function that implements a step"""

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        if action>len(self.moves_list):
            raise ValueError(f'`action` should be less than {len(self.moves_list)}')

        proba_array = np.ones(self.moves_list.shape[0])*self.proba_others
        proba_array[action] = self.proba_selected_action
        idx_opposite = action+1 if action % 2 == 0 else action-1
        proba_array[idx_opposite] = self.proba_opposite

        # Pick action and update self._state
        sampled = rd.choice(np.arange(len(proba_array)), p=proba_array)
        self._state, new_pos = self.move_in_maze(self._state, self.moves_list[sampled])

        reward = self.reward_array[new_pos]

        # Make sure episodes don't go on forever, end when reached reward position.
        if self._state == self.reward_pos:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=np.int32), reward=reward)
        else:
            return ts.transition(
                np.array([self._state], dtype=np.int32), reward=reward, discount=self.gamma)

    def move_in_maze(self, start_position, move):
        """ Returns the next cell given the selected position and move, DOES NOT change Maze2D._state, nor signals
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

    def _1d_to_2d_idx(self, idx):
        """ :returns 2d index tuple of scalar idx, useful to address the maze_array"""
        return np.where(self.maze_array == idx)

    def get_coordinates(self, idx):
        """ :returns 2d index tuple of scalar idx, useful to address for plotting"""
        return tuple(np.array(np.where(self.maze_array == idx)).squeeze())

    def build_transition_matrix(self, separate_actions=False):
        """ Builds transition matrix of current cell arrangement of the maze
        :param separate_actions: if True builds a transition matrix for each action, address them in the first dimension
        if False, just builds tot_room_number*tot_room_number matrix
        """

        if separate_actions is False:
            transition_matrix = np.zeros([self.tot_room_number, self.tot_room_number])
            for p in range(self.tot_room_number):
                for move in self.moves_list:
                    transition_matrix[p, self.move_in_maze(p, move)[0]] += 1

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
                    transition_matrix[idx_action, p, self.move_in_maze(p, action)[0]] += self.proba_selected_action
                    transition_matrix[idx_action, p, self.move_in_maze(p, opposite_action)[0]] += self.proba_opposite

                    # Loop through all other possible moves that such action can produce (with probabilities)
                    for idx_possible_move, possible_move in enumerate(self.moves_list):
                        if idx_possible_move == idx_action or idx_possible_move == idx_opposite_action:
                            continue
                        else:
                            transition_matrix[idx_action, p, self.move_in_maze(p, possible_move)[0]] += self.proba_others

            # Normalize row by row
            row_sums = transition_matrix.sum(axis=(0,2))
            transition_matrix /= row_sums[:, np.newaxis]

        return transition_matrix

    def shuffle(self, separe_actions=False):
        """ Permutation of all cell numbers of the maze, recomputes the transition matrix
        :returns new transition matrix"""

        self.maze_array = np.random.permutation(self.maze_array.reshape(-1)).reshape(self.size)
        self.transition_matrix = self.build_transition_matrix(separate_actions=separe_actions)

        return self.transition_matrix

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

    def set_reset_state(self, state):
        """Sets a specific state to put the agent into after each reset, if None, the reset is random"""
        if state is not None:
            assert state < self.tot_room_number, f"State {state} does not exist in the environment"
        self.reset_state = state

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

    def generate_spike_train(self, pos=None, actions=None):
        """ Generates a spike train matrix for the entire maze, the spike frequency is higher for the current position
            in the maze.
            If actions are specified, the spike train will encode the set of all state_action pairs available.
        :param pos: scalar indicating the state idx (use self._state if you want the current state)
        :param actions: np.array of actions indices to be coded in the spike train, or scalar.
        :returns spike_matrix of dimensions [t_steps, self.tot_room_number] or [t_steps, self.tot_room_number*tot_actions]
        organized in a way like np.shape(self.tot_room_number, tot_actions).reshape(-1)
        """
        if not self._spike_train_set:
            self._t_steps = 100
            self._high_freq_p = 1.
            self._low_freq_p = 0.

        if pos is None:
            pos = self._state

        if actions is None:
            spike_matrix_shape = np.hstack([self._t_steps, self.tot_room_number]).reshape(-1)
            spike_matrix = np.array(rd.uniform(0., 1., size=spike_matrix_shape) < self._low_freq_p, dtype=float)
            spike_matrix[:, pos] = np.array(rd.uniform(0., 1., size=self._t_steps) < self._high_freq_p, dtype=float)
        else:
            tot_actions = len(self.moves_list)
            spike_matrix_shape = np.hstack([self._t_steps, self.tot_room_number*tot_actions]).reshape(-1)
            spike_matrix = np.array(rd.uniform(0., 1., size=spike_matrix_shape) < self._low_freq_p, dtype=float)

            if np.isscalar(actions):
                actions = [actions]
            for a in actions:
                spike_matrix[:, pos+a*self.tot_room_number] = np.array(
                    rd.uniform(0., 1., size=self._t_steps) < self._high_freq_p, dtype=float)

        return spike_matrix

    def plot_t_mat(self, ax, t_mat=None):
        """Plots a transition matrix or the current transition matrix without separate actions"""
        if t_mat is None:
            t_mat = self.transition_matrix

        pcol = ax.pcolor(t_mat)
        ax.invert_yaxis()
        return pcol

    @staticmethod
    def flatten_spk_train(spk_train):
        """ Reshapes spike train output following the correct convention"""
        return spk_train.reshape(spk_train.shape[0], -1, order='F')

    @staticmethod
    def spk_train_to_tensor(spk_train, dtype=tf.float32):
        """ Returns the flattened spike train in tensor format with first dimension as the batch size, e.g.
        [1, t_steps, n_in] """
        return tf.expand_dims(tf.constant(Maze2D.flatten_spk_train(spk_train), dtype=dtype), axis=0)