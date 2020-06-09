"""
Created by Flavio Martinelli at 14:42 09/05/2020
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def base_fig(size, ax=None):
    """ Creates and returns fig, axis of the empty maze. Ticks are properly set up and each cell is size 1x1,
    the center of each cell is at coordinates x+0.5, y+0.5"""

    size = (size[1], size[0])
    fig = None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=size)

    ax.grid(True, 'major')

    ax.set_ylim([0, size[1]])
    ax.set_yticks(np.arange(size[1]), minor=False)
    ax.set_yticks(np.arange(size[1]) + 0.5, minor=True)
    ax.set_yticklabels('', minor=False)
    ax.set_yticklabels([str(n) for n in np.arange(size[1])], minor=True)
    ax.invert_yaxis()
    ax.yaxis.tick_left()

    ax.set_xlim([0, size[0]])
    ax.set_xticks(np.arange(size[0]), minor=False)
    ax.set_xticks(np.arange(size[0]) + 0.5, minor=True)
    ax.set_xticklabels('', minor=False)
    ax.set_xticklabels([str(n) for n in np.arange(size[0])], minor=True)
    ax.xaxis.tick_top()

    return fig, ax


def color_square(ax, coo, color, **kwargs):
    """ Colors a square of the maze (set alpha<1 to see better number of cell) """
    ax.add_patch(patches.Rectangle((coo[1], coo[0]),
                                    1,          # width
                                    1,          # height
                                    facecolor=color,
                                    **kwargs))


def draw_cell_numbers(ax, maze_array, **kwargs):
    """ Plots values of a 2d array of the same size of the maze """

    for x in range(maze_array.shape[1]):
        for y in range(maze_array.shape[0]):
            val = maze_array[y,x]
            ax.annotate(val,
                        xy=(x + 0.5, y + 0.5),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='gray', fontweight='bold', **kwargs)


def draw_arrow(ax, start_coo, end_coo, color, **kwargs):
    """ Draws arrow from start to end"""
    start_coo = np.array((start_coo[1], start_coo[0])) + 0.5
    end_coo = np.array((end_coo[1], end_coo[0])) + 0.5

    vec = end_coo - start_coo
    skip_vec = vec / (4 * np.linalg.norm(vec))
    start_arrow = start_coo + skip_vec
    dv_arrow = vec - 2 * skip_vec

    ax.add_patch(patches.Arrow(start_arrow[0], start_arrow[1],
                               dv_arrow[0], dv_arrow[1],
                               width=.5, facecolor=color, **kwargs))


