"""
Created by Flavio Martinelli at 15:26 07/05/2020
"""

import matplotlib.pyplot as plt
from old_scripts.maze_task import Maze2D
import maze_plot as mplot

size=(4,4)

maze = Maze2D(size=size, walls=False)
maze.agent_pos = maze.maze_array[1,2]

fig, ax = mplot.base_fig(size)
mplot.draw_cell_numbers(ax, maze.maze_array)
mplot.color_square(ax, maze.get_coordinates(maze.agent_pos), 'b')
plt.show()

maze = Maze2D(size=(4,4), walls=False)
t_mat = maze.build_transition_matrix(separate_actions=True, proba_opposite=0.15, proba_others=0.05)
